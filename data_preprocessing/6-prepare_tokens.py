import sys
import os
import pickle
import tqdm
import numpy as np
from pathlib import Path
import torch
import argparse

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        default='interx'
    )

    parser.add_argument(
        '--motion_representations',
        default='intergen_262'
    )

    args, _ = parser.parse_known_args()
    args.motion_representations = args.motion_representations.split(',')
    return args


def process_single_dataset(args):
    data_dir = Path(f'~/data/data/motion/{args.dataset}').expanduser()
    save_dir = data_dir / 'normalizers'
    save_dir.mkdir(exist_ok=True)

    for motion_representation in args.motion_representations:
        src_dir = data_dir / motion_representation
        motion_paths = src_dir.glob('*')
        actions = []
        nactions = []
        reactions = []
        for mp in tqdm.tqdm(motion_paths):
            try:
                if mp.name.endswith('pkl'):
                    with mp.open('rb') as f:
                        motion = pickle.load(f)
                    if not 32 <= len(motion['reaction']) <= 256:
                        continue
                    reactions.append(torch.from_numpy(motion['reaction']))
                    if 'action' in motion.keys():
                        actions.append(torch.from_numpy(motion['action']))
                        nactions.append(torch.from_numpy(motion['naction']))
                else:
                    with mp.open('rb') as f:
                        motion = np.load(f)
                    reactions.append(torch.from_numpy(motion))
            except Exception as e:
                print(e)

        reactions = torch.cat(reactions, dim=0)

        sd = {
            'reaction': {
                'mean': reactions.mean(0),
                'std': reactions.std(0)
            }
        }
        if actions != []:
            actions = torch.cat(actions, dim=0)
            sd.update({
                'action': {
                    'mean': actions.mean(0),
                    'std': actions.std(0)
                }
            })
            nactions = torch.cat(nactions, dim=0)
            sd.update({
                'naction': {
                    'mean': nactions.mean(0),
                    'std': nactions.std(0)
                }
            })
            all_motion = torch.cat([actions, reactions], dim=0)
            sd.update({
                'all_motion': {
                    'mean': all_motion.mean(0),
                    'std': all_motion.std(0)
                }
            })
            egocentric_motion = torch.cat([nactions, reactions], dim=0)
            sd.update({
                'egocentric_motion': {
                    'mean': egocentric_motion.mean(0),
                    'std': egocentric_motion.std(0)
                }
            })
        with (save_dir / f'{motion_representation}.pkl').open('wb') as f:
            pickle.dump(sd, f)


def process_multi_datasets(args):
    root_dir = Path(f'~/data/data/motion').expanduser()
    save_dir = root_dir / 'normalizers'
    save_dir.mkdir(exist_ok=True)

    for motion_representation in args.motion_representations:
        actions = []
        nactions = []
        reactions = []
        all_motions = []
        for dataset in args.dataset.split(','):
            dataset_dir = root_dir / dataset
            motion_dir = dataset_dir / motion_representation
            motion_paths = list(motion_dir.glob('*'))
            for mp in tqdm.tqdm(motion_paths):
                try:
                    if mp.name.endswith('pkl'):
                        with mp.open('rb') as f:
                            motion = pickle.load(f)
                        reactions.append(torch.from_numpy(motion['reaction']))
                        all_motions.append(torch.from_numpy(motion['reaction']))
                        if 'action' in motion.keys():
                            actions.append(torch.from_numpy(motion['action']))
                            nactions.append(torch.from_numpy(motion['naction']))
                            all_motions.append(torch.from_numpy(motion['naction']))
                    else:
                        with mp.open('rb') as f:
                            motion = np.load(f)
                        reactions.append(torch.from_numpy(motion))
                except Exception as e:
                    print(e)

        reactions = torch.cat(reactions, dim=0)
        all_motions = torch.cat(all_motions, dim=0)

        sd = {
            'reaction': {
                'mean': reactions.mean(0),
                'std': reactions.std(0)
            }, 
            'all_motion': {
                'mean': all_motions.mean(0),
                'std': all_motions.std(0),
            }
        }
        if actions != []:
            actions = torch.cat(actions, dim=0)
            sd.update({
                'action': {
                    'mean': actions.mean(0),
                    'std': actions.std(0)
                }
            })
            nactions = torch.cat(nactions, dim=0)
            sd.update({
                'naction': {
                    'mean': nactions.mean(0),
                    'std': nactions.std(0)
                }
            })

        with (save_dir / f'{motion_representation}.pkl').open('wb') as f:
            pickle.dump(sd, f)


if __name__ == '__main__':
    args = get_args()
    datasets = args.dataset.split(',')
    if len(datasets) == 1:
        process_single_dataset(args)
    else:
        process_multi_datasets(args)
