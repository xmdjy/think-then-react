#%% 
if __name__ == "__main__":
    import sys
    sys.path.append(sys.path[0] + r"/../../")


import tqdm
import re
import numpy as np
import random
import pickle
import torch

from src.utils import setup_logger, pad
from src.utils.motion_representation_converter import MotionRepresentationConverter
from src.utils.normalizer import TorchNormalizer
from src.datasets.dataset_base import DatasetBase


logger = setup_logger(__file__)
mrc = MotionRepresentationConverter()


class MotionCLIPDataset(DatasetBase):
    def __init__(
        self,
        dataset_dir,
        split='train',
        epoch_scaling=1,
        max_motion_length=256,
        min_motion_length=32,
        motion_representation='intergen_262',
        tiny_dataset=False,
        test_ar_correspondence='',
    ):
        super().__init__(dataset_dir=dataset_dir, split=split, epoch_scaling=epoch_scaling, tiny_dataset=tiny_dataset)
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.motion_representation = motion_representation
        self.test_ar_correspondence = test_ar_correspondence

        logger.info(f'{dataset_dir.split("/")[-1]}/{split} initializing...')

        # 0. load ids
        ids = (self.dataset_dir / 'splits' / f'{split}.txt').read_text().strip('\n').split('\n')
        if tiny_dataset:
            ids = ids[:200]

        # 1. load text and features
        logger.info('Loading texts')
        texts = {}
        valid_ids = []
        for file_id in ids:
            texts_dir = self.dataset_dir / 'texts'
            try:
                texts[file_id] = (texts_dir / f'{file_id}.txt').read_text().strip().split('\n')
                valid_ids.append(file_id)
            except:
                pass
        ids = valid_ids
        
        # 3. load normalized data
        logger.info('Loading motion data')
        self.normalizer = TorchNormalizer(
            statistics_dict=pickle.load(
                (self.dataset_dir.parent / 'normalizers' / f'{motion_representation}.pkl').open('rb')
            )
        )
        data_dict, valid_ids = self._get_data_dict(ids=ids)
        ids = valid_ids
        self.motion_dict = data_dict
        
        # 4. done
        self.ids = sorted(ids, key=lambda k: data_dict[k]['length'])
        self.texts = texts

        self.familiarity = {i+1: int(label) for i, label in enumerate((self.dataset_dir / 'annots' / 'familiarity.txt').read_text().strip().split('\n'))}

        logger.info(f'{dataset_dir.split("/")[-1]}/{split} initialization done.')
    
    def _get_data_dict(self, ids):
        data_dict = {}
        valid_ids = []
        for file_id in tqdm.tqdm(ids):
            motion_path = self.dataset_dir / self.motion_representation / f'{file_id}.pkl'
            j3d_path = self.dataset_dir / 'joints3d_22' / f'{file_id}.pkl'
            try:
                ar_test = self.test_ar_correspondence
                with motion_path.open('rb') as f:
                    data = pickle.load(f)
                if ar_test:
                    with j3d_path.open('rb') as f:
                        j3d_data = pickle.load(f)
                
                data_len = len(data['reaction'])
                if data_len > self.max_motion_length or data_len < self.min_motion_length:
                    continue

                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v)
                 
                action = data['action']
                action = self.normalizer.normalize(action, key='all_motion')

                reaction = data['reaction']
                if ar_test != '':
                    reaction_shifted = j3d_data['reaction']
                    if ar_test.startswith('pos'):
                        delta = float(ar_test[3:])
                        x = np.random.random() > 0.5
                        if x:
                            delta = np.array([np.random.choice([-delta, delta]), 0, 0])
                        else:
                            delta = np.array([0, 0, np.random.choice([-delta, delta])])
                        reaction_shifted += delta
                    elif ar_test.startswith('time'):
                        delta = int(ar_test[4:])
                        reaction_shifted = np.concatenate([reaction_shifted[delta:], reaction_shifted[-1:].repeat(delta, 0)], axis=0)
                    reaction_shifted = torch.from_numpy(mrc.convert('j3d', 'i262', reaction_shifted))
                    reaction_shifted = self.normalizer.normalize(reaction_shifted, key='all_motion')
                    reaction_shifted, boolean_mask, length = pad(reaction_shifted, length=self.max_motion_length, dim=0, value=0)

                reaction = self.normalizer.normalize(reaction, key='all_motion')
                
                action, boolean_mask, length = pad(action, length=self.max_motion_length, dim=0, value=0)
                reaction, boolean_mask, length = pad(reaction, length=self.max_motion_length, dim=0, value=0)
                label = int(re.findall(r'A(\d+)', file_id)[0])

                data_dict[file_id] = {'action': action, 'reaction': reaction, 'boolean_mask': boolean_mask, 'length': length, 'label': label}
                if ar_test:
                    data_dict[file_id].update({
                        'reaction_shifted': reaction_shifted
                    })
                valid_ids.append(file_id)
            except FileNotFoundError:
                continue
        return data_dict, valid_ids

    @property
    def real_length(self):
        return len(self.ids)

    def getitem(self, index):
        real_index = index % self.real_length
        file_id = self.ids[real_index]

        res = dict()
        res['id'] = file_id
        res.update(self.motion_dict[file_id])

        familiarity = self.familiarity[int(re.findall(r'G(\d+)', file_id)[0])]
        res.update({
            'text': random.choice(self.texts[file_id]),
            'familiarity': familiarity
        })

        rand_id = random.choice(self.ids)
        res.update({
            'random_reaction': self.motion_dict[rand_id]['reaction'],
            'random_length': self.motion_dict[rand_id]['length'],
        })

        return res

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # for d in ['interx']:
    #     for split in ['train', 'val', 'test']:
    #         ds = MotionCLIPDataset(
    #             dataset_dir=f'~/data/data/motion/{d}',
    #             split=split,
    #             tiny_dataset=True,
    #         )
    #         dl = DataLoader(ds, batch_size=32, shuffle=False)
    #         print(f'len: {next(iter(dl))["length"]}')

    for d in ['interx']:
        for ar_test in ['time10']:
            ds = MotionCLIPDataset(
                dataset_dir=f'~/data/data/motion/{d}',
                split='test',
                tiny_dataset=True,
                test_ar_correspondence=ar_test
            )
            dl = DataLoader(ds, batch_size=32, shuffle=True)
            # print(f'len: {next(iter(dl))["length"]}')
            print(next(iter(dl)))
