#%% 
if __name__ == "__main__":
    import sys
    sys.path.append(sys.path[0] + r"/../../")


import tqdm
import numpy as np
import random
import pickle
import re
import torch

from src.utils import setup_logger, pad
from src.utils.normalizer import TorchNormalizer
from src.datasets.dataset_base import DatasetBase


logger = setup_logger(__file__)


class MotionVQVAEDataset(DatasetBase):
    def __init__(
        self,
        dataset_dir,
        split='train',
        epoch_scaling=1,
        max_motion_length=256,
        min_motion_length=32,
        motion_representation='intergen_262',
        tiny_dataset=False,
        use_h3d=False,
        abs_action=False,
    ):
        super().__init__(dataset_dir=dataset_dir, split=split, epoch_scaling=epoch_scaling, tiny_dataset=tiny_dataset)
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.motion_representation = motion_representation
        self.use_h3d = use_h3d
        self.abs_action = abs_action
        if self.split != 'train':
            self.min_motion_length = 32

        logger.info(f'{dataset_dir.split("/")[-1]}/{split} initializing...')

        # 0. load ids
        ids = (self.dataset_dir / 'splits' / f'{split}.txt').read_text().strip('\n').split('\n')
        if self.tiny_dataset:
            ids = ids[:200]

        # 1. load text
        logger.info('Loading texts')
        texts_dir = self.dataset_dir / 'texts'
        texts = {}
        valid_ids = []
        for file_id in ids:
            try:
                texts[file_id] = (texts_dir / f'{file_id}.txt').read_text().strip().split('\n')
                valid_ids.append(file_id)
            except:
                pass
        self.ids = valid_ids
        self.texts = texts
        
        # 2. load normalized data
        logger.info('Loading motion data')
        self.normalizer = TorchNormalizer(
            statistics_dict=pickle.load(
                (self.dataset_dir / 'normalizers' / f'{motion_representation}.pkl').open('rb')
            )
        )

        if self.split == 'train':
            self.motions = self._load_humanml3d_motions() if self.use_h3d else []
            self.motions.extend(self._load_training_data())
        else:
            self.motion_dict, self.padded_motion_dict, self.ids = self._load_val_data()

        logger.info(f'{dataset_dir.split("/")[-1]}/{split} initialization done.')
    
    def _load_training_data(self):
        motions = []
        for file_id in tqdm.tqdm(self.ids):
            motion_path = self.dataset_dir / self.motion_representation / f'{file_id}.pkl'
            try:
                with motion_path.open('rb') as f:
                    data = pickle.load(f)
                
                data_len = len(data['reaction'])
                if data_len > self.max_motion_length or data_len < self.min_motion_length:
                    continue

                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v)
                
                reaction = self.normalizer.normalize(data['reaction'], key='all_motion')
                motions.append(reaction)
                motions.append(reaction)  # double reaction

                if self.abs_action:
                    action = self.normalizer.normalize(data['action'], key='all_motion')
                    motions.append(action)
                else:
                    naction = self.normalizer.normalize(data['naction'], key='all_motion')
                    motions.append(naction)

            except FileNotFoundError:
                continue
        return motions

    def _load_humanml3d_motions(self):
        h3d_ids = (self.dataset_dir.parent / 'humanml3d' / 'splits' / 'all.txt').read_text().strip('\n').split('\n')
        if self.tiny_dataset:
            h3d_ids = h3d_ids[:200]
        
        motions = []
        for file_id in tqdm.tqdm(h3d_ids, desc='load h3d motion'):
            motion_path = self.dataset_dir.parent / 'humanml3d' / self.motion_representation / f'{file_id}.pkl'
            try:
                with motion_path.open('rb') as f:
                    data = pickle.load(f)
                
                data_len = len(data['reaction'])
                if data_len > self.max_motion_length or data_len < self.min_motion_length:
                    continue

                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v)
                
                reaction = self.normalizer.normalize(data['reaction'], key='all_motion')
                motions.append(reaction)
            except:
                pass
        return motions

    def _load_val_data(self):
        motion_dict = {}
        padded_motion_dict = {}
        valid_ids = []
        for file_id in tqdm.tqdm(self.ids):
            motion_path = self.dataset_dir / self.motion_representation / f'{file_id}.pkl'
            try:
                with motion_path.open('rb') as f:
                    data = pickle.load(f)
                
                data_len = len(data['reaction'])
                if data_len > self.max_motion_length or data_len < self.min_motion_length:
                    continue

                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v)
                
                reaction = self.normalizer.normalize(data['reaction'], key='all_motion')
                action = self.normalizer.normalize(data['action'], key='all_motion')
                motion_dict[file_id] = {
                    'reaction': reaction,
                    'length': reaction.shape[0]
                }
                padded_action, boolean_mask, _ = pad(action, length=self.max_motion_length, dim=0, value=0)
                padded_reaction, _, _ = pad(reaction, length=self.max_motion_length, dim=0, value=0, get_boolean_mask=False)
                padded_motion_dict[file_id] = {
                    'action': padded_action, 'reaction': padded_reaction, 'boolean_mask': boolean_mask, 'label': int(re.findall(r'A(\d+)', file_id)[0])
                }
                valid_ids.append(file_id)

            except FileNotFoundError:
                continue

        return motion_dict, padded_motion_dict, valid_ids

    @property
    def real_length(self):
        if self.split == 'train':
            return len(self.motions)
        else:
            return len(self.ids)
    
    def getitem(self, index):
        if self.split == 'train':
            return self.get_train_item(index=index)
        else:
            return self.get_val_item(index=index)

    def get_train_item(self, index):
        motion = self.motions[index]
        length = len(motion)
        idx = random.randint(0, length - self.min_motion_length)
        return {
            'motion': motion[idx: idx + self.min_motion_length, :]
        }

    def get_val_item(self, index):
        file_id = self.ids[index]

        res = dict()
        res['id'] = file_id

        motion_dict = self.motion_dict[file_id]
        length = motion_dict['length']
        idx = random.randint(0, length - self.min_motion_length)
        vq_reaction = motion_dict['reaction'][idx: idx + self.min_motion_length, :]

        res.update({
            'motion': vq_reaction,
            'length': length
        })
        if self.split != 'train':
            padded_motion_dict = self.padded_motion_dict[file_id]
            res.update({
                'padded_action': padded_motion_dict['action'],
                'padded_reaction': padded_motion_dict['reaction'],
                'boolean_mask': padded_motion_dict['boolean_mask'],
                'label': padded_motion_dict['label']
            })

        res.update({'text': random.choice(self.texts[file_id])})

        return res

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    for d in ['interx']:
        for split in ['train', 'val']:
            ds = MotionVQVAEDataset(
                dataset_dir=f'~/data/data/motion/{d}',
                split=split,
                tiny_dataset=True,
                use_h3d=True
            )
            dl = DataLoader(ds, batch_size=2, shuffle=True)
            print(f'len: {len(ds), next(iter(dl))}')

# %%
