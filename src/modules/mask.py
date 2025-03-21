import torch


def get_triu_mask(h, w, device='cpu', batch_size=None, dtype=torch.float32, diagonal=1, step_length=1):
    fill_value = True if dtype == bool or dtype == torch.bool else float('-inf')
    if step_length == 1:
        mask = torch.triu(torch.full((h, w), fill_value=fill_value, dtype=dtype, device=device), diagonal=diagonal)
    elif step_length > 1:
        h_pad = ((h + step_length - 1) // step_length) * step_length
        w_pad = ((w + step_length - 1) // step_length) * step_length
        mask = torch.triu(torch.ones(h_pad // step_length, w_pad // step_length, dtype=dtype, device=device), diagonal=1)
        mask = mask.repeat_interleave(step_length, dim=0).repeat_interleave(step_length, dim=1)[:h, :w]
    else:
        raise ValueError(f'{step_length} is not a valid step_length')

    if batch_size is None:
        return mask
    else: 
        return torch.stack([mask] * batch_size, dim=0)
