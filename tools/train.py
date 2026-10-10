# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS


def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    effective_seed = int(args.seed if args.seed is not None else cfg.yaml_cfg.get('seed', 42))
    seed_source = 'cli' if args.seed is not None else 'config/fallback'
    print(f'[seed-audit] effective_seed={effective_seed} source={seed_source}', flush=True)
    output_dir = cfg.yaml_cfg.get('output_dir')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'seed_audit.txt'), 'w', encoding='utf-8') as f:
            f.write(f'effective_seed={effective_seed}\n')
            f.write(f'source={seed_source}\n')
            f.write('python_random=numpy=torch_cpu=torch_cuda=distributed_setup\n')
            f.write('dataloader_workers=PyTorch_worker_seed\n')
    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()
    


# v10 seed control
import random as _random
import numpy as _np

def _set_seed(seed: int = 42):
    import torch as _torch
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.backends.cudnn.benchmark     = False
    _torch.backends.cudnn.deterministic = True

_set_seed(42)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--stop-after-epoch', type=int, help='stop after completing this zero-based epoch while keeping the original total schedule')

    # priority 1
    parser.add_argument('-u', '--update', action='append', default=[], help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
