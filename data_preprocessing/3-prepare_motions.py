#%%  joints3d_22 to intergen_262
import sys
import os
import pickle
import numpy as np
from pathlib import Path
import torch
from concurrent.futures import ProcessPoolExecutor as PPE

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')
from src.utils.motion_representation_converter import MotionRepresentationConverter

motion_map = {
    'joints3d_22': 'j3d',
    'joints12d_22': 'j12d',
    'intergen_262': 'i262',
    'humanml3d_263': 'h263'
}


tgt_motion = 'intergen_262'
dataset = 'interx'

data_root_dir = Path(f'~/data/data/motion/{dataset}').expanduser()
src_data_dir = data_root_dir / 'joints3d_22'
save_dir = data_root_dir / tgt_motion
save_dir.mkdir(exist_ok=True)


def single_process(motion_data_path):
    try:
        mrc = MotionRepresentationConverter()
        if motion_data_path.name.endswith('pkl'):
            with motion_data_path.open('rb') as f:
                motion_dict = pickle.load(f)
            reaction = motion_dict['reaction']
            tgt_reaction = mrc('j3d', motion_map[tgt_motion], reaction)
            res = {'reaction': tgt_reaction}
            if 'action' in motion_dict:
                action = motion_dict['action']
                tgt_action = mrc('j3d', motion_map[tgt_motion], action)
                res['action'] = tgt_action

                naction = motion_dict['naction']
                tgt_naction = mrc('j3d', motion_map[tgt_motion], naction)
                res['naction'] = tgt_naction

                res['action_x'] = motion_dict['action_x']
                res['action_z'] = motion_dict['action_z']
                res['action_r'] = motion_dict['action_r']
            with (save_dir / f'{motion_data_path.stem}.pkl').open('wb') as f:
                pickle.dump(res, f)
        else:
            if (save_dir / f'{motion_data_path.stem}.npy').exists():
                return
            with motion_data_path.open('rb') as f:
                motion = np.load(f)
            tgt_reaction = mrc('j3d', motion_map[tgt_motion], motion)
            res = tgt_reaction
            with (save_dir / f'{motion_data_path.stem}.npy').open('wb') as f:
                np.save(f, res)
    except ValueError as e:
        print(e)


motion_data_path_list = list(src_data_dir.glob('*'))
# single_process(motion_data_path_list[0])
with PPE() as ppe:
    list(ppe.map(single_process, motion_data_path_list))
