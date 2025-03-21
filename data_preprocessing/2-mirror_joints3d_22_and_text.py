#%%
import sys
import numpy as np
import os
import pickle
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor as PPE

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')
from data_preprocessing.utils import mirror_joints3d_22, mirror_text


ds = 'interx'
data_dir = Path(f'~/data/data/motion/{ds}').expanduser()
joints_dir = data_dir / 'joints3d_22'
text_dir = data_dir / 'texts'
splits_dir = data_dir / 'splits'

process_text = True


def single_process(file_id):
    try:
        motion_path = joints_dir / f'{file_id}.pkl'
        motion_dict = pickle.load(motion_path.open('rb'))
        x, z, r = motion_dict['action_x'], motion_dict['action_z'], motion_dict['action_r']
        mirrored_action, mirrored_reaction, mirrored_naction =\
            mirror_joints3d_22(motion_dict['action']), mirror_joints3d_22(motion_dict['reaction']), mirror_joints3d_22(motion_dict['naction'])
        if process_text:
            texts = (text_dir / f'{file_id}.txt').read_text().split('\n')
            mirrored_texts = [mirror_text(t) for t in texts]
    except Exception as e:
        print(e)
        return None
    else:
        with (joints_dir / f'M{file_id}.pkl').open('wb') as f:
            pickle.dump(
                obj={
                    'action': mirrored_action,
                    'reaction': mirrored_reaction,
                    'naction': mirrored_naction,
                    'action_x': -x,
                    'action_z': z,
                    'action_r': r
                },
                file=f
            )
        if process_text:
            (text_dir / f'M{file_id}.txt').write_text('\n'.join(mirrored_texts))
        return f'M{file_id}'

# %%
raw_train_ids = (splits_dir / 'train.txt').read_text().strip('\n').split('\n')

with PPE() as ppe:
    new_train_ids = list(ppe.map(single_process, raw_train_ids))

if process_text:
    with (splits_dir / 'train.txt').open('a') as f:
        f.writelines([f'\n{t}' for t in new_train_ids if t is not None])

    with (splits_dir / 'all.txt').open('a') as f:
        f.writelines([f'\n{t}' for t in new_train_ids if t is not None])
