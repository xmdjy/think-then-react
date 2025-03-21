#%%
import sys
import os
import pickle
import random
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')
from src.utils.motion_representation_converter import MotionRepresentationConverter
from src.utils.plot import animate_multiple_joints3d_22


ds = 'interx'
joints_dirs = list(Path(f'~/data/data/motion/{ds}/intergen_262').expanduser().glob('*.pkl'))
text_dir = Path(f'~/data/data/motion/{ds}/texts').expanduser()
mrc = MotionRepresentationConverter()

# %%
for i in range(2):
    p = random.choice(joints_dirs)
    text = (text_dir / f'{p.stem}.txt').read_text().split('\n')[0]
    print(text)

    motion_dict = pickle.load(p.open('rb'))

    if 'inter' in ds.lower():
        action, reaction = mrc('i262', 'j3d', motion_dict['action']), mrc('i262', 'j3d', motion_dict['reaction'])
        animate_multiple_joints3d_22([action, reaction], ['b', 'g'], title=text, file_path=f'temp_{i}.mp4')
    else:
        motion = mrc('i262', 'j3d', motion_dict['reaction'])
