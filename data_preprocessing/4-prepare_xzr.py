#%%
import sys
import numpy as np
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor as PPE

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')


ds = 'interx'
data_dir = Path(f'~/data/data/motion/{ds}').expanduser()
joints_dir = data_dir / 'joints3d_22'
i262_dir = data_dir / 'intergen_262'
splits_dir = data_dir / 'splits'

face_joint_indx = [2, 1, 17, 16]
r_hip, l_hip, sdr_r, sdr_l = face_joint_indx


def get_xzr(motion):
    x, z, r = [], [] , []
    for m in motion:
        x.append(m[0, 0])
        z.append(m[0, 2])
        across = m[r_hip] - m[l_hip]
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        r.append(np.arctan2(forward_init[0, 0], forward_init[0, 2]))
    return np.array(x), np.array(z), np.array(r)


def single_process(file_id):
    try:
        j3d_path = joints_dir / f'{file_id}.pkl'
        j3d_dict = pickle.load(j3d_path.open('rb'))

        i262_path = i262_dir / f'{file_id}.pkl'
        i262_dict = pickle.load(i262_path.open('rb'))

        reaction_xzr = get_xzr(j3d_dict['reaction'])
        action_xzr = get_xzr(j3d_dict['action'])

        j3d_dict['reaction_x'] = reaction_xzr[0]
        j3d_dict['reaction_z'] = reaction_xzr[1]
        j3d_dict['reaction_r'] = reaction_xzr[2]
        j3d_dict['action_x'] = action_xzr[0]
        j3d_dict['action_z'] = action_xzr[1]
        j3d_dict['action_r'] = action_xzr[2]

        i262_dict['reaction_x'] = reaction_xzr[0]
        i262_dict['reaction_z'] = reaction_xzr[1]
        i262_dict['reaction_r'] = reaction_xzr[2]
        i262_dict['action_x'] = action_xzr[0]
        i262_dict['action_z'] = action_xzr[1]
        i262_dict['action_r'] = action_xzr[2]

    except Exception as e:
        print(e)
        return None
    else:
        with j3d_path.open('wb') as f:
            pickle.dump(
                obj=j3d_dict,
                file=f
            )
        with i262_path.open('wb') as f:
            pickle.dump(
                obj=i262_dict,
                file=f
            )

# %%
ids = (splits_dir / 'all.txt').read_text().strip('\n').split('\n')

with PPE() as ppe:
    list(ppe.map(single_process, ids))
