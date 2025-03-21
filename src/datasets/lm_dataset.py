#%% 
if __name__ == "__main__":
    import sys
    sys.path.append(sys.path[0] + r"/../../")


import re
import copy
import tqdm
import numpy as np
import random
import pickle
import torch

from src.utils import setup_logger, pad
from src.utils.normalizer import TorchNormalizer
from src.datasets.dataset_base import DatasetBase
from src.utils.motion_representation_converter import MotionRepresentationConverter
from src.utils.constants import VALUE_RANGES, INTERX_LABEL_MAPPING, INTERX_FAMILIARITY_MAPPING


logger = setup_logger(__file__)
mrc = MotionRepresentationConverter()


class LMDataset(DatasetBase):
    def __init__(
        self,
        dataset_dir,
        vqvae_ckpt_path,
        motion_token_template='<motion_{}>',
        x_template='<pos_x_{}>',
        z_template='<pos_z_{}>',
        r_template='<rot_r_{}>',
        n_x_bins=100,
        n_z_bins=100,
        n_r_bins=16,
        use_h3d=False,
        stage='pretrain',
        split='train',
        epoch_scaling=1,
        max_motion_length=256,
        min_motion_length=32,
        motion_representation='intergen_262',
        tiny_dataset=False,
        sample_mode=False,
    ):
        super().__init__(dataset_dir=dataset_dir, split=split, epoch_scaling=epoch_scaling, tiny_dataset=tiny_dataset)
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.motion_representation = motion_representation
        self.sample_mode = sample_mode

        self.motion_token_template = motion_token_template
        self.vqvae_ckpt_path = vqvae_ckpt_path
        self.n_x_bins = n_x_bins
        self.n_z_bins = n_z_bins
        self.n_r_bins = n_r_bins
        self.x_template = x_template
        self.z_template = z_template
        self.r_template = r_template
        self.use_h3d = use_h3d
        self.stage = stage
        self.abs_action = 'abs' in vqvae_ckpt_path

        self.x_index = mrc.tokenize_value(value_range=VALUE_RANGES['x'], num_bins=n_x_bins, value=0)
        self.z_index = mrc.tokenize_value(value_range=VALUE_RANGES['z'], num_bins=n_z_bins, value=0)
        self.r_index = mrc.tokenize_value(value_range=VALUE_RANGES['r'], num_bins=n_r_bins, value=0)

        logger.info(f'{dataset_dir.split("/")[-1]}/{split} initializing...')

        # 0. load ids
        ids = (self.dataset_dir / 'splits' / f'{split}.txt').read_text().strip('\n').split('\n')
        if self.tiny_dataset:
            ids = ids[:200]

        # 1. load text
        logger.info('Loading texts')
        texts_dir = self.dataset_dir / 'texts'
        caption_list_dict = {}
        valid_ids = []
        for file_id in ids:
            try:
                caption_list_dict[file_id] = (texts_dir / f'{file_id}.txt').read_text().strip().split('\n')
                valid_ids.append(file_id)
            except:
                pass
        self.ids = valid_ids
        self.caption_list_dict = caption_list_dict
        
        # 2. load normalized data
        logger.info('Loading motion data')
        self.normalizer = TorchNormalizer(
            statistics_dict=pickle.load(
                (self.dataset_dir / 'normalizers' / f'{motion_representation}.pkl').open('rb')
            )
        )
        # 3. load h3d data
        if use_h3d:
            self.h3d_caption_list_dict, self.h3d_data_dict, self.h3d_ids = self._load_h3d_data()

        # 4. load h-h group data
        self.familiarity = {i+1: int(label) for i, label in enumerate((self.dataset_dir / 'annots' / 'familiarity.txt').read_text().strip().split('\n'))}

        # 5. load lm data
        if self.abs_action:
            self.lm_data_dict, self.ids = self._load_abs_lm_data()
        else:
            self.lm_data_dict, self.ids = self._load_lm_data()
        if self.split != 'train' or self.tiny_dataset or self.sample_mode:
            self.padded_motion_dict, self.ids = self._load_motion_generation_data()

        logger.info(f'{dataset_dir.split("/")[-1]}/{split} initialization done.')
    
    def get_m_list(self, motion_list):
        return [self.motion_token_template.format(t) for t in motion_list]
        
    def get_m_x_z_r_token_list(self, motion_list, x_list, z_list, r_list):
        length = len(x_list) // 4 * 4
        motion_token_list = [self.motion_token_template.format(t) for t in motion_list]
        x_token_list = []
        z_token_list = []
        r_token_list = []
        for x, z, r in list(zip(x_list, z_list, r_list))[0: length: 4]:
            x_token_list.append(self.x_template.format(mrc.tokenize_value(VALUE_RANGES['x'], self.n_x_bins, x)))
            z_token_list.append(self.z_template.format(mrc.tokenize_value(VALUE_RANGES['z'], self.n_z_bins, z)))
            r_token_list.append(self.r_template.format(mrc.tokenize_value(VALUE_RANGES['r'], self.n_r_bins, r)))

        assert len(motion_token_list) == len(x_token_list)
        return motion_token_list, x_token_list, z_token_list, r_token_list

    def _load_h3d_data(self):
        dataset_dir = self.dataset_dir.parent / 'humanml3d'
        ids = (dataset_dir / 'splits' / 'all.txt').read_text().strip('\n').split('\n')
        if self.tiny_dataset:
            ids = ids[:200]
        
        texts_dir = dataset_dir / 'texts'
        caption_list_dict = {}
        valid_ids = []
        for file_id in ids:
            try:
                caption_list_dict[file_id] = (texts_dir / f'{file_id}.txt').read_text().strip().split('\n')
                valid_ids.append(file_id)
            except:
                pass

        h3d_data_dict = {}
        final_ids = []
        for file_id in tqdm.tqdm(valid_ids):
            token_dict_path = dataset_dir / self.vqvae_ckpt_path.replace('/', '__slash__') / f'{file_id}.pkl'
            try:
                with token_dict_path.open('rb') as f:
                    data_dict = pickle.load(f)
                token_length = len(data_dict['reaction'])

                reaction_motion = ''.join([self.motion_token_template.format(t) for t in data_dict['reaction']])

                h3d_data_dict[file_id] = {
                    'h3d_motion': reaction_motion,
                    'h3d_token_length': token_length,
                }
                final_ids.append(file_id)
            except FileNotFoundError:
                continue
        print(len(final_ids))
        return caption_list_dict, h3d_data_dict, final_ids

    def _load_lm_data(self):
        lm_data_dict = {}
        valid_ids = []
        for file_id in tqdm.tqdm(self.ids):
            token_dict_path = self.dataset_dir / self.vqvae_ckpt_path.replace('/', '__slash__') / f'{file_id}.pkl'
            try:
                with token_dict_path.open('rb') as f:
                    data_dict = pickle.load(f)
                token_length = len(data_dict['reaction'])
                half_length = token_length // 2

                reaction_motion_list, reaction_x_list, reaction_z_list, reaction_r_list = self.get_m_x_z_r_token_list(
                    data_dict['reaction'], data_dict['reaction_x'], data_dict['reaction_z'], data_dict['reaction_r']
                )
                reaction_xzr_list = [x + z + r for x, z, r in zip(reaction_x_list, reaction_z_list, reaction_r_list)]
                reaction_chars = ''.join(chr(token_id) for token_id in data_dict['reaction'])

                action_motion_list, action_x_list, action_z_list, action_r_list = self.get_m_x_z_r_token_list(
                    data_dict['naction'], data_dict['action_x'], data_dict['action_z'], data_dict['action_r']
                )
                action_xzr_list = [x + z + r for x, z, r in zip(action_x_list, action_z_list, action_r_list)]

                label_idx = int(re.findall(r'A(\d+)', file_id)[0])
                familiarity_level = self.familiarity[int(re.findall(r'G(\d+)', file_id)[0])]

                lm_data_dict[file_id] = {
                    # for sampling in getitem
                    'reaction_motion_list':         ','.join(reaction_motion_list),
                    'reaction_xzr_list':       ','.join(reaction_xzr_list),
                    # placeholders
                    'reaction_xzr':            ','.join(reaction_xzr_list),
                    'reaction_initial_xzr':       reaction_xzr_list[0],
                    'reaction_mid_xzr':       reaction_xzr_list[half_length],
                    'reaction_motion':         ''.join(reaction_motion_list),
                    'reaction_initial_motion': reaction_motion_list[0],
                    'reaction_mid_motion':     reaction_motion_list[half_length],
                    'reaction_motion_p1':      ''.join(reaction_motion_list[ :half_length]),
                    'reaction_motion_p2':      ''.join(reaction_motion_list[half_length: ]),
                    'reaction_chars':      reaction_chars,

                    # for sampling in getitem
                    'action_motion_list':         ','.join(action_motion_list),
                    'action_xzr_list':       ','.join(action_xzr_list),
                    # placeholders
                    'action_xzr':            ','.join(action_xzr_list),
                    'action_initial_xzr':       action_xzr_list[0],
                    'action_mid_xzr':        action_xzr_list[half_length],
                    'action_motion':         ''.join(action_motion_list),
                    'action_initial_motion': action_motion_list[0],
                    'action_mid_motion':     action_motion_list[half_length],
                    'action_motion_p1':      ''.join(action_motion_list[ :half_length]),
                    'action_motion_p2':      ''.join(action_motion_list[half_length: ]),

                    'label_idx': label_idx,
                    'label': INTERX_LABEL_MAPPING[label_idx],

                    'familiarity_idx': familiarity_level,
                    'familiarity': INTERX_FAMILIARITY_MAPPING[familiarity_level],

                    'token_length': token_length,
                }
                valid_ids.append(file_id)
            except FileNotFoundError:
                continue
        return lm_data_dict, valid_ids

    def _load_motion_generation_data(self):
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
                
                reaction = self.normalizer.normalize(torch.from_numpy(data['reaction']), key='all_motion')
                action = self.normalizer.normalize(torch.from_numpy(data['action']), key='all_motion')
                length = len(reaction) // 4 * 4
                reaction = reaction[:length, ]
                action = action[:length, ]

                padded_action, boolean_mask, _ = pad(action, length=self.max_motion_length, dim=0, value=0)
                padded_reaction, _, _ = pad(reaction, length=self.max_motion_length, dim=0, value=0, get_boolean_mask=False)
                padded_motion_dict[file_id] = {
                    'action': padded_action, 'reaction': padded_reaction, 'boolean_mask': boolean_mask, 'length': length
                }
                valid_ids.append(file_id)

            except FileNotFoundError:
                continue

        return padded_motion_dict, valid_ids

    @property
    def real_length(self):
        return len(self.ids)
    
    def getitem(self, index):
        file_id = self.ids[index]

        data = copy.deepcopy(self.lm_data_dict[file_id])

        data['id'] = file_id
        label = data['label']
        caption_list = self.caption_list_dict[file_id]
        # caption_list = [f'Label: {label}. Description: {c}' for c in caption_list]
        data['caption'] = random.choice(caption_list)

        if 'familiarity' in data:
            data['familiarity_caption'] = f'The relationship of the two people is: {data["familiarity"]}'

        data['all_captions'] = '\t'.join(caption_list)

        if self.split != 'train' or self.tiny_dataset or self.sample_mode:
            data.update(self.padded_motion_dict[file_id])
        
        if self.abs_action:
            return data

        token_length = data['token_length']
        half_length = token_length // 2
        begin_idx = random.randint(0, half_length)
        end_idx = begin_idx + half_length

        reaction_motion_list = data['reaction_motion_list'].split(',')
        reaction_xzr_list = data['reaction_motion_list'].split(',')

        action_motion_list = data['action_motion_list'].split(',')
        action_xzr_list = data['action_motion_list'].split(',')

        data.update({
            'reaction_motion_clip': ''.join(reaction_motion_list[begin_idx: end_idx]),
            'reaction_initial_motion_clip': reaction_motion_list[begin_idx],
            'reaction_initial_xzr_clip': reaction_xzr_list[begin_idx],

            'action_motion_clip': ''.join(action_motion_list[begin_idx: end_idx]),
            'action_initial_motion_clip': action_motion_list[begin_idx],
            'action_initial_xzr_clip': action_xzr_list[begin_idx],
        })

        human = 'reaction' if random.random() > 0.5 else 'action'
        data.update({
            'human_motion': data[f'{human}_motion'],
            'human_initial_motion': data[f'{human}_initial_motion'],
            'human_xzr': data[f'{human}_xzr'],
            'human_initial_xzr': data[f'{human}_initial_xzr'],
        })
        
        if self.use_h3d:
            h3d_index = random.randrange(0, len(self.h3d_ids))
            h3d_file_id = self.h3d_ids[h3d_index]
            h3d_data = self.h3d_data_dict[h3d_file_id]
            h3d_caption_list = self.h3d_caption_list_dict[h3d_file_id]
            data.update(h3d_data)
            data.update({
                'h3d_caption': random.choice(h3d_caption_list),
                'h3d_all_captions': '\t'.join(h3d_caption_list)
            })
        
        if self.stage == 'finetune':
            m2t_motion_ratio = random.choice([0.25, 0.5, 1])
            m2t_motion = ''.join(action_motion_list[:int(m2t_motion_ratio * token_length)])
            data['ft_m2t_motion'] = m2t_motion
        return data


#%%
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    for d in ['interx']:
        for split in ['train', 'val']:
            ds = LMDataset(
                dataset_dir=f'~/data/data/motion/{d}',
                split=split,
                tiny_dataset=True,
                vqvae_ckpt_path='logs/motion_vqvae/motion_vqvae/20241129-140049_bs256_trained/checkpoints/epoch40__step74333__monitor-0.207.ckpt',
                # use_h3d=True,
            )
            dl = DataLoader(ds, batch_size=2, shuffle=True)
            item = next(iter(dl))
            for k, v in item.items():
                print(f'{k}: {v}')

# %%
