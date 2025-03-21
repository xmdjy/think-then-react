from typing import Dict
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .model_base import ModelBase
from ..modules.embeddings import PositionalEncoding
from ..utils.constants import MOTION_REPRESENTATION_INFO, TEXT_FEATURE_INFO
from ..metrics.common import calculate_diversity, euclidean_distance_matrix, calculate_top_k, calculate_fid


class MotionCLIP(ModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        self.motion_representation = model_kwargs.motion_representation
        self.motion_representation_info = MOTION_REPRESENTATION_INFO[model_kwargs.motion_representation]
        self.motion_feature_size = motion_feature_size = self.motion_representation_info['feature_size']
        self.output_size = output_size = model_kwargs.output_size
        self.n_labels = n_labels = model_kwargs.n_labels

        # text
        text_feature_name = model_kwargs.text_feature_name
        self.text_feature_info = TEXT_FEATURE_INFO[text_feature_name]
        text_model = transformers.CLIPTextModel.from_pretrained(text_feature_name)
        self.text_emb = copy.deepcopy(text_model.text_model.embeddings).eval()
        for p in self.text_emb.parameters():
            p.requires_grad_(False)
        del text_model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(text_feature_name)
        self.text_feature_size = text_feature_size = self.text_feature_info['feature_size']
        self.text_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=text_feature_size,
                nhead=model_kwargs.n_heads,
                activation='gelu',
                batch_first=True,
                dropout=model_kwargs.dropout
            ),
            num_layers=model_kwargs.n_encoder_layers,
        )
        self.text_linear_out = nn.Linear(text_feature_size, output_size)

        # motion
        self.action_mask_coef = model_kwargs.get('action_mask_coef', 0)
        input_motion_size = (motion_feature_size - 4) * 2 if self.action_mask_coef >=0 else (motion_feature_size - 4)
        self.motion_lin_in = nn.Linear(input_motion_size, text_feature_size)
        self.motion_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=text_feature_size,
                nhead=model_kwargs.n_heads,
                activation='gelu',
                batch_first=True,
                dropout=model_kwargs.dropout
            ),
            num_layers=model_kwargs.n_encoder_layers,
        )

        # classification head
        self.motion_linear_out = nn.Linear(text_feature_size, output_size)
        if model_kwargs.cls_weight > 0:
            self.motion_cls_head = nn.Sequential(
                nn.Dropout(model_kwargs.dropout),
                nn.Linear(output_size, output_size),
                nn.Tanh(),
                nn.Dropout(model_kwargs.dropout),
                nn.Linear(output_size, n_labels)
            )

        self.pe = PositionalEncoding(d_model=text_feature_size, dropout=model_kwargs.dropout)
        self.cls = torch.nn.Parameter(torch.zeros(size=(1, 1, text_feature_size)))
        torch.nn.init.normal_(self.cls, mean=0, std=0.02)

        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.latent_scale = torch.nn.Parameter(torch.Tensor([model_kwargs.get('init_latent_scale', 1)]))
    
    def combine_motion(self, reaction, action):
        seq_length = reaction.shape[1]

        reaction = reaction[:, :, :-4]
        action = action[:, :, :-4]
        if self.action_mask_coef > 0:
            action_mask = [1] + [0] * self.action_mask_coef
            action_mask = action_mask * (seq_length // len(action_mask) + 1)
            action_mask = torch.tensor(action_mask[:seq_length], device=reaction.device)
            action = torch.einsum('bsh,s->bsh', action, action_mask)
        
        if self.action_mask_coef >= 0:
            motion = torch.cat([reaction, action], dim=-1)
        else:
            motion = reaction

        return motion

    def encode_motion(self, reaction, action, boolean_mask):
        motion = self.combine_motion(reaction=reaction, action=action)
        bsz, seq_length, _ = reaction.shape

        encoder_input = self.motion_lin_in(motion)
        encoder_input = torch.concat([self.cls.expand(size=(bsz, 1, self.text_feature_size)), encoder_input], dim=1)

        encoder_input = self.pe(encoder_input)
        src_pad_mask = torch.cat(
            [torch.zeros(size=(bsz, 1), dtype=motion.dtype, device=motion.device), boolean_mask], dim=1
        )

        encoder_output = self.motion_transformer.forward(src=encoder_input, src_key_padding_mask=src_pad_mask)[:, 0, :]
        motion_feature = self.motion_linear_out(encoder_output)
        return motion_feature

    def encode_text(self, text_list):
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = ~inputs.attention_mask.to(device=self.device, dtype=bool)

        with torch.no_grad():
            hidden_states = self.text_emb(input_ids)

        last_hidden_state = self.text_transformer.forward(
            src=hidden_states,
            src_key_padding_mask=attention_mask
        )

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.tokenizer.eos_token_id)
            .int()
            .argmax(dim=-1),
        ]
        text_feature = self.text_linear_out(pooled_output)
        return text_feature

    def get_log_dict(self, batch, batch_idx, split) -> Dict:
        boolean_mask = batch[f'boolean_mask']
        reaction = batch['reaction']
        action = batch['action']
        text_list = batch['text']

        if 'label' in batch:
            labels = batch['label']

        motion_embeddings = self.encode_motion(reaction=reaction, action=action, boolean_mask=boolean_mask)
        text_embeddings = self.encode_text(text_list=text_list)

        logits_per_motion = self.latent_scale.exp() * motion_embeddings @ text_embeddings.t()

        logits_per_d = logits_per_motion.t()
        batch_size = motion_embeddings.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=motion_embeddings.device)
        ce_from_motion_loss = self.ce_loss(logits_per_motion, ground_truth)
        ce_from_d_loss = self.ce_loss(logits_per_d, ground_truth)

        res = {
            f'{split}/d_ce': ce_from_d_loss,
            f'{split}/motion_ce': ce_from_motion_loss,
        }

        total_loss = (ce_from_motion_loss + ce_from_d_loss) / 2
        if self.model_kwargs.cls_weight > 0:
            logits = self.motion_cls_head(motion_embeddings)
            ce_from_cls_loss = self.ce_loss(input=logits, target=labels)
            res[f'{split}/cls_ce'] = ce_from_cls_loss
            total_loss = ce_from_cls_loss * self.model_kwargs.cls_weight + total_loss

        res[f'{split}/total_loss'] = total_loss

        return res
        
    def get_metrics(self, batch, split, shift=False, q=False):
        log_dict = {}

        boolean_mask = batch[f'boolean_mask']
        action = batch['action']
        reaction = batch['reaction']
        if shift or q:  # ignore this when training your own model
            if q:
                reaction_shifted = batch['reaction_shifted']
            else:
                reaction_shifted = batch['random_reaction']
            min_lengthes = torch.stack([batch['length'], batch['random_length']], dim=-1).min(-1).values
            shifted_boolean_mask = boolean_mask.clone().detach()
            for b in range(min_lengthes.shape[0]):
                shifted_boolean_mask[b, min_lengthes[b]:] = True
            
            motion_embeddings = self.encode_motion(reaction=reaction, action=action, boolean_mask=boolean_mask)
            shifted_motion_embeddings = self.encode_motion(reaction=reaction_shifted, action=action, boolean_mask=shifted_boolean_mask)
            fid = calculate_fid(motion_embeddings.detach().cpu().numpy(), shifted_motion_embeddings.detach().cpu().numpy())
            reaction = reaction_shifted
            mask = shifted_boolean_mask
        else:
            fid = 0.0
        log_dict[f'{split}/fid'] = fid

        text_list = batch['text']
        labels = batch['label']

        motion_embeddings = self.encode_motion(reaction=reaction, action=action, boolean_mask=boolean_mask)

        if self.model_kwargs.cls_weight > 0:
            logits = self.motion_cls_head(motion_embeddings)
            acc_1 = (logits.argmax(-1) == labels).sum() / labels.shape[0]
            log_dict[f'{split}/acc_1'] = acc_1

            _, top5_preds = torch.topk(logits, 5, dim=1)
            top5_correct = (labels.unsqueeze(1) == top5_preds).any(dim=1).float()
            acc_5 = top5_correct.sum() / labels.shape[0]
            log_dict[f'{split}/acc_5'] = acc_5

        motion_embeddings = motion_embeddings.detach().cpu().numpy()

        text_embeddings = self.encode_text(text_list=text_list).detach().cpu().numpy()
        batch_size = boolean_mask.shape[0]

        gt_dist_mat = euclidean_distance_matrix(text_embeddings, motion_embeddings)
        mm_dist = gt_dist_mat.trace() / batch_size
        log_dict[f'{split}/mm_dist'] = mm_dist

        argsmax = np.argsort(gt_dist_mat, axis=1)
        top_k_mat = calculate_top_k(argsmax, top_k=3)
        r_prec = top_k_mat.sum(axis=0) / batch_size

        total_ranking = 0
        for i in range(3):
            log_dict[f'{split}/top {i + 1}'] = r_prec[i]
            total_ranking += r_prec[i]

        log_dict['monitor'] = total_ranking
        if self.model_kwargs.cls_weight > 0:
            log_dict['monitor'] += acc_1

        div = calculate_diversity(activations=motion_embeddings, diversity_times=batch_size - 1)
        log_dict[f'{split}/div'] = div

        for k, v in log_dict.items():
            try:
                log_dict[k] = torch.tensor(v).to(self.device)
            except: pass
            
        return log_dict

    def extra_validation_step(self, batch, batch_idx=None) -> Dict:
        return self.get_metrics(batch, 'val')

    def test_step(self, batch, batch_idx=None):
        res = self.get_metrics(batch, 'test')
        self.log_dict(res, sync_dist=True)
        return res
