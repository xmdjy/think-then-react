#%% smpl to joints3d_22
from typing import List
import sys
import copy
import os
import pickle
import numpy as np
from pathlib import Path
import torch
import tqdm
from concurrent.futures import ProcessPoolExecutor as PPE
import argparse

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')
from third_party.HumanML3D.human_body_prior.body_model.body_model import BodyModel
from src.utils.motion_representation_converter import MotionRepresentationConverter
from data_preprocessing.utils import normalize_dual_joints3d_22, normalize_single_joints3d_22


args = None
trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0]])
mrc = MotionRepresentationConverter()

#%%
args = argparse.ArgumentParser()
args.add_argument(
    '--dataset',
    default='interx',
)

args.add_argument(
    '--devices',
    default='1'
)
args = args.parse_args()

#%%
dataset = 'Inter-X_Dataset' # if args is None else args.dataset
devices = '1' if args is None else args.devices
devices = [torch.device(f'cuda:{i}') for i in devices.split(',')]


def single_process_smpl_to_joint3d_22(smpl_paths: List[Path], pose_save_dir, body_models_path, interaction_order, device, fps_downsample_rate, length_range, n_joints=22):
    def get_pose_seq_np(person, down_sample_rate):
        data = {
            'trans': torch.Tensor(person['trans'][::down_sample_rate, ...]).to(device),
            'pose_body': torch.Tensor(person['pose_body'][::down_sample_rate, ...]).view(-1, 21 * 3).to(device),
            'root_orient': torch.Tensor(person['root_orient'][::down_sample_rate, ...]).to(device),
        }
        pose_seq_np = body_model(**data).Jtr.detach().cpu().numpy()
        if dataset == 'interhuman':
            pose_seq_np = np.dot(pose_seq_np, trans_matrix)
        return pose_seq_np

    body_model = BodyModel(bm_fname=body_models_path).to(device)

    for dual_smpl_path in tqdm.tqdm(smpl_paths):
        file_id = dual_smpl_path.stem
        # if (pose_save_dir / f'{file_id}.pkl').exists():
        #     continue
        try:
            # interx
            if dual_smpl_path.is_dir():
                path_1 = dual_smpl_path / 'P1.npz'
                path_2 = dual_smpl_path / 'P2.npz'
                with path_1.open('rb') as f1, path_2.open('rb') as f2:
                    person1 = np.load(f1)
                    person2 = np.load(f2)
                    n_frames = len(person1['pose_body'])
                    if n_frames < length_range[0] or n_frames > length_range[1]:
                        continue
                    if interaction_order[file_id] == 0:
                        person1, person2 = person2, person1
                    
                    action = get_pose_seq_np(person1, down_sample_rate=fps_downsample_rate)[:, :n_joints, :]
                    reaction = get_pose_seq_np(person2, down_sample_rate=fps_downsample_rate)[:, :n_joints, :]
                    action, reaction = mrc.norm_dual_joints3d_22(action, reaction)
                    naction, (x, z, r) = mrc.norm_joint3d_22(action)
                    pose_data = {
                        'naction': naction,
                        'action': action,
                        'reaction': reaction,
                        'action_x': x,
                        'action_z': z,
                        'action_r': r,
                    }
            else:
                # interhuman
                with dual_smpl_path.open('rb') as f:
                    dual_smpl = pickle.load(f)
                n_frames = dual_smpl['frames']
                if n_frames < length_range[0] or n_frames > length_range[1]:
                    continue
                person1 = dual_smpl['person1']
                person2 = dual_smpl['person2']
                if interaction_order[file_id] == 0:
                    person1, person2 = person2, person1

                action = get_pose_seq_np(person1, down_sample_rate=fps_downsample_rate)[:, :n_joints, :]
                reaction = get_pose_seq_np(person2, down_sample_rate=fps_downsample_rate)[:, :n_joints, :]
                action, reaction = normalize_dual_joints3d_22(action, reaction)
                naction = normalize_single_joints3d_22(action)
                pose_data = {
                    'naction': naction,
                    'action': action,
                    'reaction': reaction
                }

            with (pose_save_dir / f'{file_id}.pkl').open('wb') as f:
                pickle.dump(pose_data, f)
        except Exception as e:
            print(f'{dual_smpl_path}: {e}')


if __name__ == '__main__':
    data_root_dir = Path(f'~/Think-Then-React/data/{dataset}').expanduser()
    smpl_dir = data_root_dir / 'motions'
    smpl_paths = [p for p in smpl_dir.glob('*') if p.is_dir()]
    if smpl_paths == []:
        smpl_paths = [p for p in smpl_dir.glob('*.pkl')]

    pose_save_dir = data_root_dir / 'joints3d_22'
    pose_save_dir.mkdir(exist_ok=True)

    n_proc = len(devices)

    src_fps = 60 if dataset.lower() == 'interhuman' else 120  # interx
    tgt_fps = 20
    fps_downsample_rate = src_fps // tgt_fps
    length_range = [32 * fps_downsample_rate, 256 * fps_downsample_rate]

    body_models_path = os.path.expanduser('~/data/pretrained_models/motion/body_models/smplx/SMPLX_NEUTRAL.npz')

    try:
        with (data_root_dir / 'annots' / 'interaction_order.pkl').open('rb') as f:
            interaction_order = pickle.load(f)
    except:
        interaction_order = None

    smpl_path_chunks = [
        smpl_paths[i::n_proc] for i in range(n_proc)
    ]
    single_process_smpl_to_joint3d_22(smpl_paths, pose_save_dir, body_models_path, interaction_order, devices[0], fps_downsample_rate, length_range)
    with PPE(max_workers=n_proc) as ppe:
        list(ppe.map(
            single_process_smpl_to_joint3d_22,
            smpl_path_chunks,
            [pose_save_dir] * n_proc,
            [body_models_path] * n_proc,
            [interaction_order] * n_proc,
            devices,
            [fps_downsample_rate] * n_proc,
            [length_range] * n_proc
        ))

# %%
