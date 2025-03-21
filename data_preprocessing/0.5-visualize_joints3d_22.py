#%%
import sys
import numpy as np
import os
import pickle
import random
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')
from src.utils.plot import animate_multiple_joints3d_22

ds = 'interx'
joints_dirs = list(Path(f'~/data/data/motion/{ds}/joints3d_22').expanduser().glob('*'))
text_dir = Path(f'~/data/data/motion/{ds}/texts').expanduser()

# %%
for i in range(1):
    p = random.choice(joints_dirs)
    text = (text_dir / f'{p.stem}.txt').read_text().strip().split('\n')

    if 'inter' in ds.lower():
        dual_person = pickle.load(p.open('rb'))
        action, reaction, naction = dual_person['action'], dual_person['reaction'], dual_person['naction']
        # animate_multiple_joints3d_22([action, reaction, naction], ['r', 'g', 'b'], title=text[0], file_path=f'temp_{ds}_{i}.mp4', show_axis=True)
        animate_multiple_joints3d_22([action, reaction], ['r', 'g'], title=text[0], file_path=f'temp_{ds}_{i}.mp4', show_axis=True)
    else:
        motion = pickle.load(p.open('rb'))['reaction']
        animate_multiple_joints3d_22([motion], ['b'], title=text[0], file_path=f'temp_{ds}_{p.stem}.mp4', show_axis=True)

# %%
