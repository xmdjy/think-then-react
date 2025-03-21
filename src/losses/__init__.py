import torch
import torch.nn.functional as F


def get_masked_seq2seq_loss(gt_seq, pred_seq, loss_mask, loss_mask_sum=None, loss_fn = F.smooth_l1_loss):
    shape = gt_seq.shape
    if loss_mask_sum == None:
        loss_mask_sum = loss_mask.sum()
    gt_seq = gt_seq.reshape(shape[0], shape[1], -1)
    pred_seq = pred_seq.reshape(shape[0], shape[1], -1)
    loss = loss_fn(pred_seq, gt_seq, reduction='none').mean(-1)
    loss = loss_mask * loss
    return torch.sum(loss) / loss_mask_sum
