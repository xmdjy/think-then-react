import time
import copy
import math
from typing import Dict
from collections import OrderedDict
import random
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .model_base import ModelBase
from .modeling_t5 import T5ForConditionalGeneration, T5Config
from ..modules.mask import get_triu_mask
from ..metrics.motion_generation import MotionGenerationEvaluator
from ..metrics.nlg import NLGEvaluator
from ..utils.utils import get_model_and_config_from_ckpt_path
from ..utils.log import PickleLogger, JsonLogger


class LMReactiveMotionGenerator(ModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        # vqvae
        vqvae, vqvae_config = get_model_and_config_from_ckpt_path(model_kwargs.vqvae_ckpt_path)
        del vqvae.evaluator
        for p in vqvae.parameters():
            p.requires_grad_(False)
        self.vqvae = vqvae
        self.vqvae_config = vqvae_config
        n_code = vqvae_config.model.model_kwargs.nb_code

        # lm
        lm_name = model_kwargs.lm
        config = T5Config.from_pretrained(lm_name)
        lm = T5ForConditionalGeneration.from_pretrained(lm_name).train()
        tokenizer = transformers.AutoTokenizer.from_pretrained(lm_name)
        tokenizer.add_tokens(
            [all_config.dataset.motion_token_template.format(i) for i in range(n_code)] + \
            [all_config.dataset.x_template.format(x) for x in range(model_kwargs.n_x_bins)] + \
            [all_config.dataset.z_template.format(z) for z in range(model_kwargs.n_z_bins)] + \
            [all_config.dataset.r_template.format(r) for r in range(model_kwargs.n_r_bins)] 
        )
        lm.resize_token_embeddings(len(tokenizer))
        self.lm = lm
        self.tokenizer = tokenizer

        self.stage = model_kwargs.stage
        self.abs_action = 'abs' in model_kwargs.vqvae_ckpt_path
        self.all_tasks = json.load(open(f'src/configs/{"abs_" if self.abs_action else ""}lm_tasks.json', 'r'))
        self.training_tasks = list(self.all_tasks[self.stage].keys())
        self.task_sample_weights = [1] * len(self.training_tasks)
        self.unit_size = model_kwargs.get('unit_size', 1)
        self.non_causal_token_id = self.tokenizer.encode(':')[1]

        # we use lazy loading here as 'self.device' and 'self.logger' are not initialized before training
        ## evaluator
        self._mg_evaluator = None
        self._nlg_evaluator = None
        ## log
        self._pkl_logger = None
        self._json_logger = None

        if p := model_kwargs.pretrained_path:
            state_dict = torch.load(p, map_location='cpu')['state_dict']
            target_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('lm.'):
                    target_state_dict[k] = v
            incompatible_keys = self.load_state_dict(target_state_dict, strict=False)
        
        self.test_samples = {}
        self.normalizer = None
    
    @property
    def mg_evaluator(self):
        if self._mg_evaluator is None:
            self._mg_evaluator = MotionGenerationEvaluator(self.model_kwargs.evaluator_ckpt_path, self.device)
        return self._mg_evaluator

    @property
    def nlg_evaluator(self):
        if self._nlg_evaluator is None:
            self._nlg_evaluator = NLGEvaluator(self.device)
        return self._nlg_evaluator
    
    @property
    def json_logger(self):
        if self._json_logger is None:
            self._json_logger = JsonLogger(self)
        return self._json_logger

    @property
    def pkl_logger(self):
        if self._pkl_logger is None:
            self._pkl_logger = PickleLogger(self)
        return self._pkl_logger
    
    def get_src_tgt(self, batch, split, task_name=None):
        tasks = self.all_tasks[self.stage]
        bsz = len(batch['id'])

        src_list = []
        tgt_list = []
        for b in range(bsz):
            if split == 'train':
                if self.model_kwargs.use_adaptive_sampling:
                    task_name = random.choices(self.training_tasks, self.task_sample_weights, k=1)[0]
                else:
                    task_name = random.choice(self.training_tasks)
            src_placeholders = tasks[task_name]['src_placeholders']
            tgt_placeholders = tasks[task_name]['tgt_placeholders']
            src = random.choice(tasks[task_name]['src'])
            tgt = tasks[task_name]['tgt']
            for src_ph in src_placeholders:
                src = src.replace(f'<{src_ph}>', batch[src_ph][b])
            for tgt_ph in tgt_placeholders:
                tgt = tgt.replace(f'<{tgt_ph}>', batch[tgt_ph][b])
            src_list.append(src)
            tgt_list.append(tgt)
        return src_list, tgt_list

    def get_log_dict(self, batch, batch_idx, split) -> Dict:
        # if self.training and self.stage == 'finetune' and batch_idx == 0:
        #     self.log_dict(self.get_mg_metrics(batch=batch, batch_idx=batch_idx, split=split))

        log_dict = dict()

        if split == 'train':
            src, tgt = self.get_src_tgt(batch=batch, split=split)
            log_dict[f'{split}/total_loss'] = self.get_lm_loss(src, tgt)
        elif split == 'val':
            for idx, task_name in enumerate(self.training_tasks):
                src, tgt = self.get_src_tgt(batch=batch, task_name=task_name, split=split)
                self.task_sample_weights[idx] = log_dict[f'{split}/{task_name}_loss'] = self.get_lm_loss(src, tgt)
            log_dict[f'{split}/total_loss'] = sum(log_dict.values())

        return log_dict

    def get_non_causal_length(self, token_indices):
        non_causal_index = (token_indices == self.non_causal_token_id).nonzero().squeeze(-1)[-1].item()  # find <:> token
        return non_causal_index + 2  # there is a < > after <:> so plus 2
    
    def get_causal_masks(self, bsz, src_length, tgt_length, src_ids):
        if self.stage == 'pretrain':
            return None, None

        src_mask = torch.ones(size=(bsz, src_length, src_length), device=self.device)
        memory_mask = torch.ones(size=(bsz, tgt_length, src_length), device=self.device)
        for b in range(bsz):
            try:
                non_causal_length = self.get_non_causal_length(token_indices=src_ids[b])
            except:  # no causal token <:> found
                src_mask[b, :, :] = memory_mask[b, :, :] = 0
                continue
            causal_length = min(src_length - non_causal_length, tgt_length)
            src_mask[b, :, :non_causal_length] = 0
            memory_mask[b, :, :non_causal_length] = 0
            triu = get_triu_mask(causal_length, causal_length, device=self.device, dtype=bool, diagonal=self.unit_size)
            src_mask[b, non_causal_length: non_causal_length+causal_length, non_causal_length:  non_causal_length+causal_length] = triu
            memory_mask[b, :causal_length, non_causal_length:  non_causal_length+causal_length] = triu
        src_mask = src_mask * torch.finfo(torch.float32).min
        memory_mask = memory_mask * torch.finfo(torch.float32).min
        return src_mask.unsqueeze(1), memory_mask.unsqueeze(1)
    
    def get_lm_loss(self, src, tgt):
        src_encoding = self.tokenizer(src, padding='longest', truncation=True, return_tensors='pt')
        tgt_encoding = self.tokenizer(tgt, padding='longest', truncation=True, return_tensors='pt')

        src_ids = src_encoding.input_ids.to(self.device)
        src_mask = src_encoding.attention_mask.to(self.device)
        tgt_ids = tgt_encoding.input_ids.to(self.device)
        tgt_mask = tgt_encoding.attention_mask.to(self.device)
        tgt_ids[tgt_ids == 0] = -100  # ignore_index=-100

        bsz, src_length = src_ids.shape[:2]
        tgt_length = tgt_ids.shape[1]
        decoder_input_ids = self.lm._shift_right(tgt_ids)
        if self.training:
            masking_mask = torch.rand_like(decoder_input_ids, dtype=torch.float32) > 1 - self.model_kwargs.mask_ratio
            decoder_input_ids[masking_mask] = 0

        src_causal_mask, memory_causal_mask = self.get_causal_masks(
            bsz=bsz, src_length=src_length, tgt_length=tgt_length, src_ids=src_ids
        )

        res = self.lm.forward(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            labels=tgt_ids,
            decoder_attention_mask=tgt_mask,
            src_causal_mask=src_causal_mask,
            memory_causal_mask=memory_causal_mask
        )
        return res.loss

    def extra_validation_step(self, batch, batch_idx=None) -> Dict:
        log_dict = dict()
        if self.stage == 'pretrain':
            return log_dict

        log_dict.update(self.get_mg_metrics(batch, 'val', batch_idx))
        # log_dict.update(self.get_nlg_metrics(batch, 'val', batch_idx))
        log_dict['monitor'] = -log_dict['mg/fid']  # if mg in eval_task, replace monitor
        return log_dict

    def test_step(self, batch, batch_idx=None, eval_nlg_only=False, use_gt_prompt=False, nlg_generation_config={}, eval_nlg_action_ratio=1):
        log_dict = dict()
        if eval_nlg_only:
            log_dict.update(self.get_nlg_metrics(batch, 'test', batch_idx, nlg_generation_config=nlg_generation_config, action_ratio=eval_nlg_action_ratio))
        else:
            log_dict.update(self.get_mg_metrics(batch, 'test', batch_idx, use_gt_prompt=use_gt_prompt))
        self.log_dict(log_dict, sync_dist=True)
        return log_dict

    def get_nlg_metrics(self, batch, split, batch_idx=None, nlg_generation_config={}, action_ratio=1):
        ref_sentences = batch['all_captions']
        for i, ref in enumerate(ref_sentences):
            ref_sentences[i] = ref.split('\t')

        pred_sentences = self.generate_caption(batch, generation_configs=nlg_generation_config, action_ratio=action_ratio)

        try:
            res = self.nlg_evaluator.evaluate(pred_sentences=pred_sentences, reference_sentences=ref_sentences)

            if self.global_rank == 0 and batch_idx == 0:
                json_logs = copy.deepcopy(res)
                json_logs.update({
                    'epoch': self.current_epoch,
                    'sentences': [{'ref': ref, 'pred': pred} for ref, pred in zip(ref_sentences, pred_sentences)],
                })
                self.json_logger.log(json_logs)
        except Exception as e:
            print(f'error detected during validation: {e}\npred_sentences: {pred_sentences}\nref_list:{ref_sentences}\nbatch_idx:{batch_idx}')
        
        for k, v in res.items():
            try:
                res[k] = torch.tensor(v).to(self.device)
            except: pass
        return res

    def generate_caption(self, batch, src_list=None, generation_configs={}, action_ratio=1):
        if src_list is None:
            template = self.all_tasks['eval_nlg']
            src_placeholders = template['src_placeholders']
            src_list = []
            bsz = len(batch['id'])

            src_list = []
            for b in range(bsz):
                src = random.choice(template['src'])
                for src_ph in src_placeholders:
                    if src_ph == 'action_motion':
                        action_motion = batch['action_motion_list'][b].split(',')
                        action_motion = ''.join(action_motion[:int(len(action_motion) * action_ratio)])
                        src = src.replace('<action_motion>', action_motion)
                    else:
                        src = src.replace(f'<{src_ph}>', batch[src_ph][b])
                
                src_list.append(src)

        src_encoding = self.tokenizer(src_list, padding='longest', truncation=True, return_tensors='pt')
        src_ids = src_encoding.input_ids.to(self.device)
        src_mask = src_encoding.attention_mask.to(self.device) 
        encoder_outputs = self.lm.encoder.forward(input_ids=src_ids, attention_mask=src_mask)
        res = self.lm.generate(encoder_outputs=encoder_outputs, do_sample=True, **generation_configs)
        res = self.tokenizer.batch_decode(res, skip_special_tokens=True)
        return res

    def get_mg_metrics(self, batch, split, batch_idx=None, use_gt_prompt=False):
        ids = batch['id']
        gt_actions = batch['action']
        gt_reactions = batch['reaction']
        lengths = batch['length']
        captions = batch['caption']
        labels = batch['label_idx']
        boolean_mask = batch['boolean_mask']

        pred_reaction, _ = self.generate_reaction(batch, use_gt_prompt=use_gt_prompt)
        res = self.mg_evaluator.evaluate(
            gt_action = gt_actions,
            gt_reaction = gt_reactions,
            pred_reaction = pred_reaction,
            boolean_mask = boolean_mask,
            text_list = captions,
            labels = labels,
        )

        for k, v in res.items():
            try:
                res[k] = torch.tensor(v).to(self.device)
            except: pass
            
        return res
    
    @torch.no_grad()
    def generate_reaction(self, batch, nlg_generation_configs={}, use_gt_prompt=False):
        # Better understand this method with chatbots, even the authors theirselves

        def get_reaction_generation_prompt(batch, captions):
            template = self.all_tasks['eval_mg']
            src_placeholders = self.all_tasks['eval_mg']['src_placeholders']

            src_list = []
            bsz = len(batch['id'])

            src_list = []
            for b in range(bsz):
                src = random.choice(template['src'])
                for src_ph in src_placeholders:
                    if src_ph == 'caption':
                        src = src.replace('<caption>', captions[b])
                    else:
                        src = src.replace(f'<{src_ph}>', batch[src_ph][b])
                src_list.append(src)
            return src_list

        def think(batch, action_motions):
            template = self.all_tasks['pretrain']['action-to-caption']
            src_placeholders = template['src_placeholders']

            src_list = []
            bsz = len(batch['id'])

            src_list = []
            for b in range(bsz):
                src = random.choice(template['src'])
                for src_ph in src_placeholders:
                    if src_ph == 'action_motion':
                        src = src.replace(f'<{src_ph}>', action_motions[b])
                    src = src.replace(f'<{src_ph}>', batch[src_ph][b])
                src_list.append(src)
            pred_captions = self.generate_caption(batch=batch, src_list=src_list, generation_configs=nlg_generation_configs)
            return pred_captions

        lengths = batch['token_length']
        bsz = len(lengths)
        max_length = math.ceil(max(lengths) / self.unit_size) + 1
        if use_gt_prompt:
            captions = batch['caption']
        else:
            captions = ['The other person reacts to the action' for _ in range(bsz)]

        action_motion_str_list = batch['action_motion_list']
        action_motion_lists = []
        for action_motion_str in action_motion_str_list:
            action_motion_tokens = action_motion_str.split(',')
            action_motion_lists.append(action_motion_tokens)
        
        tgt_ids = torch.zeros(size=(len(lengths), 1), dtype=torch.int64, device=self.device)

        predicted_captions = [[] for _ in range(bsz)]
        for input_length in range(1, max_length + 1):
            action_motions = []
            for b, action_length in enumerate(lengths):
                action_motions.append(''.join(action_motion_lists[b][:min(action_length, input_length)]))

            if not use_gt_prompt and input_length % self.model_kwargs.rethinking_interval == 0:
                captions = think(batch, action_motions=action_motions)
                for b in range(bsz):
                    predicted_captions[b].append(captions[b])
            prompts = get_reaction_generation_prompt(batch, captions=captions)

            src = []
            for b in range(bsz):
                src.append(prompts[b] + action_motions[b])

            src_encoding = self.tokenizer(src, padding='longest', truncation=True, return_tensors='pt', add_special_tokens=False)
            src_ids = src_encoding.input_ids.to(self.device)
            pred_ids = self.lm.generate(inputs=src_ids, decoder_input_ids=tgt_ids, max_new_tokens=1, do_sample=True)
            tgt_ids = torch.cat([tgt_ids, pred_ids[:, -1:]], dim=-1)

        output_strings = self.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
        return self.decode_reaction_string(batch, output_strings), predicted_captions

    def decode_reaction_string(self, batch, output_strings):
        pred_reaction = []
        gt_action = batch['action']
        gt_reaction = batch['reaction']
        bsz, seq_length, motion_feature_size = gt_action.shape
        for b, s in enumerate(output_strings):
            gt_token_length = batch['token_length'][b]
            tokens = [int(num) for num in re.findall(f'<motion_(\d+)', s)][:gt_token_length]
            if len(tokens) == 0:
                tokens = [0]
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            pred_motion = self.vqvae.decode(tokens, first_frame=gt_reaction[b:b+1, 0, :]).squeeze()
            len_pred_motion = len(pred_motion)
            if len_pred_motion < seq_length:
                pred_motion = torch.cat(
                    [pred_motion, torch.zeros(size=(seq_length - len_pred_motion, motion_feature_size), device=self.device)],
                    dim=0
                )
            elif len_pred_motion > seq_length:
                pred_motion = pred_motion[:seq_length]
            pred_reaction.append(pred_motion)
        pred_reaction = torch.stack(pred_reaction, dim=0)

        return pred_reaction
