#%%
import sys
import os
import pickle
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt


tgt_motion = 'intergen_262'
dataset = 'interx'

data_root_dir = Path(f'~/data/data/motion/{dataset}').expanduser()
src_data_dir = data_root_dir / 'joints3d_22'

x, z, r = [], [] , []
for p in src_data_dir.glob('*.pkl'):
    m = pickle.load(p.open('rb'))
    x.append(m['action_x'])
    z.append(m['action_z'])
    r.append(m['action_r'])
    x.append(m['reaction_x'])
    z.append(m['reaction_z'])
    r.append(m['reaction_r'])

x = np.concatenate(x)
z = np.concatenate(z)
r = np.concatenate(r)

#%%
eps = 1e-4
print(f'x: [{min(x) - eps}, {max(x) + eps}]')
print(f'z: [{min(z) - eps}, {max(z) + eps}]')
print(f'r: [{min(r) - eps}, {max(r) + eps}]')
# %%
def visualize_distribution(data, num_bins=100):
    data_min = np.min(data)
    data_max = np.max(data)
    bin_width = (data_max - data_min) / num_bins
    bins = np.arange(data_min, data_max + bin_width, bin_width)
    bin_indices = np.digitize(data, bins)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xticks(bins)
    plt.show()

#%%
visualize_distribution(x, 100)
visualize_distribution(z, 100)
visualize_distribution(r, 100)
# %%
