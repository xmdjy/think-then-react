import torch
import numpy as np
import Levenshtein

from ..utils.utils import get_model_and_config_from_ckpt_path
from .common import calculate_fid, euclidean_distance_matrix, calculate_top_k, calculate_diversity


class MotionGenerationEvaluator(torch.nn.Module):
    def __init__(self, ckpt_path: str, device='cpu'):
        super().__init__()
        model, _ = get_model_and_config_from_ckpt_path(ckpt_path)
        self.model = model.to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def calculate_fid(self, gt_embeddings, pred_embeddings):
        return {
            'mg/fid': calculate_fid(gt_embeddings, pred_embeddings)
        }

    def calculate_div(self, motion_embeddings, diversity_times):
        return {
            'mg/div': calculate_diversity(motion_embeddings, diversity_times)
        }

    def calculate_ranking_and_mm_dist(self, text_embeddings, motion_embeddings):
        res = {}
        gt_dist_mat = euclidean_distance_matrix(text_embeddings, motion_embeddings)
        mm_dist = gt_dist_mat.trace() / text_embeddings.shape[0]
        res['mg/mm_dist'] = mm_dist

        argsmax = np.argsort(gt_dist_mat, axis=1)
        top_k_mat = calculate_top_k(argsmax, top_k=3)
        r_prec = top_k_mat.sum(axis=0) / text_embeddings.shape[0]

        for i in range(3):
            res[f'mg/top {i + 1}'] = r_prec[i]

        return res
    
    def calculate_acc(self, logits, labels):
        log_dict = dict()
        acc = (logits.argmax(-1) == labels).sum() / labels.shape[0]
        log_dict['mg/acc_1'] = acc.item()

        _, top5_preds = torch.topk(logits, 5, dim=1)
        top5_correct = (labels.unsqueeze(1) == top5_preds).any(dim=1).float()
        top5_accuracy = top5_correct.sum() / labels.shape[0]
        log_dict['mg/acc_5'] = top5_accuracy.item()
        return log_dict
    
    # def calculate_edit_distance(self, gt_chars_list, pred_chars_list, clustering=True):
    #     eds = []
    #     for g, p in zip(gt_chars_list, pred_chars_list):
    #         if clustering:
    #             eds.append(Levenshtein.distance(chars_clustering(g), chars_clustering(p)) / len(g))
    #         else:
    #             eds.append(Levenshtein.distance(g, p) / len(g))
    #     return {'mg/ed': np.mean(eds)}

    def evaluate(self, gt_action, gt_reaction, pred_reaction, boolean_mask, text_list, labels=None):
        gt_motion_embeddings = self.model.encode_motion(reaction=gt_reaction, action=gt_action, boolean_mask=boolean_mask).cpu().numpy()
        pred_motion_embeddings = self.model.encode_motion(reaction=pred_reaction, action=gt_action, boolean_mask=boolean_mask)

        if labels is not None:
            logits = self.model.motion_cls_head(pred_motion_embeddings)

        pred_motion_embeddings = pred_motion_embeddings.cpu().numpy()
        text_embeddings = self.model.encode_text(text_list).cpu().numpy()

        results = {}
        results.update(self.calculate_ranking_and_mm_dist(text_embeddings, pred_motion_embeddings))
        results.update(self.calculate_div(pred_motion_embeddings, pred_motion_embeddings.shape[0] - 1))
        results.update(self.calculate_fid(gt_motion_embeddings, pred_motion_embeddings))

        if labels is not None:
            results.update(self.calculate_acc(logits, labels))

        return results
