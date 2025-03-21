import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from .model_base import ModelBase
from ..metrics.motion_generation import MotionGenerationEvaluator
from ..modules.resnet import Res1DEncoder, Res1DDecoder
from ..utils.constants import MOTION_REPRESENTATION_INFO


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity


class VQVAEModule(nn.Module):
    def __init__(
        self,
        model_kwargs,
        nb_code=1024,
        code_dim=512,
        output_emb_width=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        motion_feature_size=262,
        with_first_frame=False,
    ):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.with_first_frame = with_first_frame
        self.quant = model_kwargs.quantizer
        self.encoder = Res1DEncoder(motion_feature_size, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Res1DDecoder(motion_feature_size, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer = QuantizeEMAReset(nb_code, code_dim, model_kwargs.mu)
        if with_first_frame:
            self.first_frame_proj = nn.Linear(motion_feature_size, output_emb_width)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        
        ## quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        if self.with_first_frame:
            first_frame_features = self.first_frame_proj(x[:, 0, :]).unsqueeze(2)
            x_quantized = x_quantized + first_frame_features

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    def forward_decoder(self, x, first_frame=None):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        if self.with_first_frame:
            first_frame_features = self.first_frame_proj(first_frame).unsqueeze(2)
            x_d = x_d + first_frame_features
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class MotionVQVAE(ModelBase):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__(model_kwargs=model_kwargs, training_kwargs=training_kwargs, all_config=all_config)

        self.motion_representation = model_kwargs.motion_representation
        self.motion_representation_info = MOTION_REPRESENTATION_INFO[model_kwargs.motion_representation]
        motion_feature_size = self.motion_representation_info['feature_size']

        self.unit_size = model_kwargs.down_t ** 2
        self.with_first_frame = model_kwargs.get('with_first_frame', False)

        self.vqvae = VQVAEModule(
            model_kwargs=model_kwargs,
            nb_code=model_kwargs.nb_code,
            code_dim= model_kwargs.code_dim,
            output_emb_width= model_kwargs.output_emb_width,
            down_t= model_kwargs.down_t,
            stride_t= model_kwargs.stride_t,
            width= model_kwargs.width,
            depth=model_kwargs.depth,
            dilation_growth_rate=model_kwargs.dilation_growth_rate,
            activation=model_kwargs.vq_act,
            norm=model_kwargs.vq_norm,
            motion_feature_size=motion_feature_size,
            with_first_frame=self.with_first_frame
        )
        # metrics
        try:
            self.evaluator = MotionGenerationEvaluator(model_kwargs.evaluator_ckpt_path).eval()
        except:
            self.evaluator = None

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def decode(self, x, first_frame=None):
        x_out = self.vqvae.forward_decoder(x, first_frame=first_frame)
        return x_out

    def get_log_dict(self, batch, batch_idx, split):
        motion = batch['motion']
        pred_motion, commit_loss, ppl = self.vqvae(motion)

        # recons
        recons_loss = F.smooth_l1_loss(motion, pred_motion)

        # vel
        gt_pos = motion[:, :, slice(*self.motion_representation_info['key_to_range']['pos'])]
        pred_pos = pred_motion[:, :, slice(*self.motion_representation_info['key_to_range']['pos'])]
        vel_loss = F.smooth_l1_loss(pred_pos[:, 1:, :] - pred_pos[:, :-1, :], gt_pos[:, 1:, :] - gt_pos[:, :-1, :])

        loss_kwargs = self.training_kwargs.loss_kwargs
        total_loss = recons_loss +  loss_kwargs.commit_weight * commit_loss +  loss_kwargs.vel_weight * vel_loss

        return {
            f'{split}/total_loss': total_loss,
            f'{split}/recon': recons_loss,
            f'{split}/commit': commit_loss,
            f'{split}/vel': vel_loss,
            f'{split}/ppl': ppl,
        }

    def extra_validation_step(self, batch, batch_idx=None):
        metrics = self.get_metrics(batch, 'val')
        metrics['monitor'] = -metrics['mg/fid']
        return metrics

    def test_step(self, batch, batch_idx=None):
        res = self.get_metrics(batch, 'test')
        self.log_dict(res, sync_dist=True)
        return res
    
    def get_metrics(self, batch, split='val'):
        gt_action = batch['padded_action']
        gt_reaction = batch['padded_reaction']
        length = batch['length']

        bsz, seq_length, motion_feature_size = gt_reaction.shape
        device, dtype = gt_reaction.device, gt_reaction.dtype

        pred_reaction = []
        for b in range(bsz):
            motion_length = length[b]
            motion_length = motion_length // (self.unit_size) * self.unit_size
            unpadded_reaction = gt_reaction[b: b+1, :motion_length, :]
            unconstructed_reaction = self.vqvae.forward(unpadded_reaction)[0]
            batch['boolean_mask'][b: b+1, motion_length:, ...] = True
            if unpadded_reaction.shape[1] != seq_length:
                unconstructed_reaction = torch.cat(
                    [unconstructed_reaction, torch.zeros(size=(1, seq_length - motion_length, motion_feature_size), device=device, dtype=dtype)],
                    dim=1
                )
            pred_reaction.append(unconstructed_reaction)
        pred_reaction = torch.cat(pred_reaction, 0)

        res = self.evaluator.evaluate(
            gt_action = gt_action,
            gt_reaction = gt_reaction,
            pred_reaction = pred_reaction,
            boolean_mask = batch['boolean_mask'],
            text_list = batch['text'],
            labels = batch['label']
        )
        return res
