import sys
import os
import pickle
import random
import tqdm
import numpy as np
from pathlib import Path
import torch
import argparse

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')
from src.utils.normalizer import TorchNormalizer
from src.utils.motion_representation_converter import MotionRepresentationConverter
from src.utils.utils import get_model_and_config_from_ckpt_path
from src.utils.plot import animate_multiple_joints3d_22


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ckpt_path',
        default='./path/to/vqvae.ckpt'
    )

    parser.add_argument(
        '--dataset',
        default='interx'
    )

    parser.add_argument(
        '--n_samples',
        default=20
    )

    parser.add_argument(
        '--device',
        type=str,
        default=1
    )

    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.device}')
    return args


@torch.no_grad()
def main():
    args = get_args()
    model, model_config = get_model_and_config_from_ckpt_path(args.ckpt_path)
    model = model.to(args.device)

    mrc = MotionRepresentationConverter()

    data_dir = Path(f'~/data/data/motion/{args.dataset}').expanduser()
    src_dir = data_dir / model_config.model.model_kwargs.motion_representation
    tgt_dir = data_dir / args.ckpt_path.replace('/', '__slash__')

    normalizer_dict = pickle.load((data_dir / 'normalizers' / f'{model_config.model.model_kwargs.motion_representation}.pkl').open('rb'))
    normalizer = TorchNormalizer(normalizer_dict)

    src_motion_paths = random.choices(list(src_dir.glob('*.pkl')), k=args.n_samples)

    for mp in tqdm.tqdm(src_motion_paths):
        try:
            with mp.open('rb') as f:
                gt_motion_dict = pickle.load(f)
            with (tgt_dir / mp.name).open('rb') as f:
                tokens_dict = pickle.load(f)

            gt_reaction = gt_motion_dict['reaction']
            gt_naction = gt_motion_dict['naction']
            pred_reaction_tokens = tokens_dict['reaction']
            pred_naction_tokens = tokens_dict['naction']

            pred_reaction = normalizer.denormalize(
                model.decode(pred_reaction_tokens.unsqueeze(0).to(args.device)),
                key='all_motion'
            ).squeeze().cpu().numpy()
            pred_naction = normalizer.denormalize(
                model.decode(pred_naction_tokens.unsqueeze(0).to(args.device)),
                key='all_motion'
            ).squeeze().cpu().numpy()

            animate_multiple_joints3d_22(
                motions=[
                    mrc.convert('i262', 'j3d', gt_reaction[:pred_reaction.shape[0], ...]),
                    mrc.convert('i262', 'j3d', pred_reaction)
                ],
                colors=['r', 'g'],
                title='reaction',
                file_path=f'temp_reaction_{mp.stem}.mp4'
            )
            animate_multiple_joints3d_22(
                motions=[
                    mrc.convert('i262', 'j3d', gt_naction[:pred_naction.shape[0], ...]),
                    mrc.convert('i262', 'j3d', pred_naction)
                ],
                colors=['r', 'g'],
                title='naction',
                file_path=f'temp_naction_{mp.stem}.mp4'
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            print(e)


if __name__ == '__main__':
    main()
