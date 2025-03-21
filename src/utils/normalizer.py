from typing import Dict, List, Union
import numpy as np
import torch


class TorchNormalizer():
    def __init__(self, statistics_dict: Dict[str, Dict[str, torch.Tensor]]):
        self.statistics_dict = statistics_dict
        for k, v in statistics_dict.items():
            for kk, vv in v.items():
                vv.requires_grad_(False)
    
    def normalize(self, data: Union[torch.Tensor, np.ndarray], key: str):
        if self.statistics_dict == None:
            return data

        if isinstance(data, np.ndarray):
            is_np_input = True
            data = torch.from_numpy(data)
        else:
            is_np_input = False

        mean = self.statistics_dict[key]['mean'].to(data.device)
        std = self.statistics_dict[key]['std'].to(data.device)
        res = (data - mean) / std
        res = torch.nan_to_num(res, nan=0.0)  # TODO: change to fill with mean

        return res.cpu().numpy() if is_np_input else res

    def denormalize(self, data: Union[torch.Tensor, np.ndarray], key: str):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            is_np_input = True
        else:
            is_np_input = False

        mean = self.statistics_dict[key]['mean'].to(data.device)
        std = self.statistics_dict[key]['std'].to(data.device)
        res = data * std + mean

        return res.cpu().numpy() if is_np_input else res
    
    def norm_batch(self, batch: Dict[str, torch.Tensor], keys: List[str] = None, device=torch.device('cuda:0')):
        if keys is None:
            keys = batch.keys()
        
        raw_device = batch[keys[0]].device  # assume all tensor values are on the same device 

        if raw_device != torch.device('cpu'):
            device = raw_device
        # else if raw_device is cpu, move data to cuda:0

        for k in keys:
            batch[k] = self.normalize(data=batch[k].to(device), key=k).to(raw_device)
        return batch
    
    def denorm_batch(self, batch: Dict[str, torch.Tensor], device=torch.device('cuda:0')):
        if keys is None:
            keys = batch.keys()
    
        raw_device = batch[keys[0]].device

        if raw_device != torch.device('cpu'):
            device = raw_device
        # else if raw_device is cpu, move data to cuda:0

        for k in keys:
            batch[k] = self.denormalize(data=batch[k].to(device), key=k).to(raw_device)
        return batch

    def norm_list_dict(self, data: List[Dict[str, torch.Tensor]], keys: List[str] = None, device=torch.device('cuda:0')):
        if keys is None:
            keys = data.keys()
        
        data_length = len(data)
        big_batch = {k:[] for k in keys}
        for k in keys:
            for idx in range(data_length):
                big_batch[k].append(data[idx][k])
            big_batch[k] = torch.stack(big_batch[k])
        
        big_batch = self.norm_batch(batch=big_batch, keys=keys, device=device)

        for idx in range(data_length):
            for k in keys:
                data[idx][k] = big_batch[k][idx]

        return data
    
    def denorm_list_dict(self, data: Dict[str, torch.Tensor], device=torch.device('cuda:0')):
        if keys is None:
            keys = data.keys()
        
        data_length = len(data)
        big_batch = {k:[] for k in keys}
        for k in keys:
            for idx in range(data_length):
                big_batch[k].append(data[idx][k])
            big_batch[k] = torch.stack(big_batch[k])
        
        big_batch = self.denorm_batch(batch=big_batch, keys=keys, device=device)

        for idx in range(data_length):
            for k in keys:
                data[idx][k] = big_batch[k][idx]

        return data
