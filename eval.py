import os
import argparse
import shutil
import tqdm
from collections import defaultdict
import lightning.pytorch as pl
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import torch
import pprint
from torch.utils.data import DataLoader

from src.utils import get_obj_from_str, instantiate_from_config, setup_logger, get_metric_statistics, dict_to_device
from src.metrics.motion_generation import MotionGenerationEvaluator


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = None


@torch.no_grad()
def evaluate(args, config, model, test_dataloader):
    nlg_generation_config = {'max_new_tokens': 200} if args.eval_nlg_only else {}
    final_res = defaultdict(list)
    final_statistics = defaultdict(list)
    for i in range(args.replication_times):
        pl.seed_everything(config.seed + i)

        results = defaultdict(list)
        for batch in tqdm.tqdm(test_dataloader):
            batch = dict_to_device(batch, device=model.device)
            step_results = model.test_step(
                batch, use_gt_prompt=args.use_gt_prompt, 
                eval_nlg_only=args.eval_nlg_only,
                eval_nlg_action_ratio=args.eval_nlg_action_ratio,
                nlg_generation_config=nlg_generation_config
            )
            for k, v in step_results.items():
                results[k].append(v)
        
        for k, v in results.items():
            final_res[k].append(torch.stack(v).cpu().numpy().mean())
        print(final_res)

    for k, v in final_res.items():
        mean, conf_interval = get_metric_statistics(v, replication_times=args.replication_times)
        final_statistics[k] = f'{mean:.3f},{conf_interval:.3f}'

    logger.info(f'evaluation results: {final_statistics}')


def get_args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='logs/model_name/dataset_name/signature_trained/checkpoints/best.ckpt'
    )

    parser.add_argument(
        '--device',
        type=int,
        default=1
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--replication_times',
        type=int,
        default=1
    )

    parser.add_argument(
        '--rethinking_interval',
        type=int,
        default=None
    )

    parser.add_argument(
        '--evaluator_ckpt_path',
        type=str,
        default=None
    )

    parser.add_argument(
        '--eval_nlg_only',
        action='store_true'
    )

    parser.add_argument(
        '--eval_nlg_action_ratio',
        type=float,
        default=1
    )

    parser.add_argument(
        '--use_gt_prompt',
        default=False,
    )

    args = parser.parse_args()

    args.ckpt_path = Path(args.ckpt_path)
    log_dir = args.ckpt_path.parent.parent

    args.device = torch.device(args.device)

    args.log_file_path = str((log_dir / 'results.log').expanduser())

    config = OmegaConf.load(log_dir / 'hparams.yaml').all_config

    return args, config


def main():
    global logger
    args, config = get_args_and_config()

    logger = setup_logger(__file__, log_file=args.log_file_path)

    logger.info(f'\n-----------------------------------------------\n')
    logger.info(f'Evaluating with ckpt: {args.ckpt_path}')
    logger.info(f'Evaluation config: {pprint.pformat(config)}')

    model_cls = get_obj_from_str(config.model.target)
    model = model_cls.load_from_checkpoint(str(args.ckpt_path), map_location=args.device, strict=False).eval()

    if p := args.evaluator_ckpt_path:
        model._mg_evaluator = MotionGenerationEvaluator(ckpt_path=p, device=args.device)

    if isinstance(args.rethinking_interval, int):
        model.model_kwargs.rethinking_interval = args.rethinking_interval 

    # load val data
    test_dataset = instantiate_from_config(config.dataset, extra_kwargs={'split': 'test'})
    model.normalizer = test_dataset.normalizer

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, persistent_workers=True, drop_last=True)

    evaluate(args=args, config=config, model=model, test_dataloader=test_dataloader)


if __name__ == '__main__':
    main()
