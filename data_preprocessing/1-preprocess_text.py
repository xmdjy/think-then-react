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


dataset = 'interx'
data_root_dir = Path(f'~/data/data/motion/{dataset}').expanduser()
text_dir = data_root_dir / 'texts'

def single_process(text_path: Path):
    try:
        texts = text_path.read_text().strip('\n').split('\n')
        target_texts = []
        for text in texts:
            target_texts.append(
                text.strip(' \n,\t').replace(
                'his/her', 'his').replace('him/her', 'him').replace('he/she', 'he').replace(
                'counter-clockwise', 'counterclockwise').replace('counter clockwise', 'counterclockwise').replace(
                'anti-clockwise', 'counterclockwise').replace('anti clockwise', 'counterclockwise')
            )
        text_path.write_text('\n'.join(target_texts))
    except ValueError as e:
        print(e)

text_path_list = list(text_dir.glob('*.txt'))
with PPE() as ppe:
    list(ppe.map(single_process, text_path_list))
