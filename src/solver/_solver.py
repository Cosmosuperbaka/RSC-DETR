# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 
from typing import Dict
import atexit
import re

from ..misc import dist_utils
from ..core import BaseConfig, YAMLConfig


def to(m: nn.Module, device: str):
    if m is None:
        return None 
    return m.to(device) 


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg 

    def _setup(self, ):
        """Avoid instantiating unnecessary classes 
        """
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cfg.model
        
        # NOTE (lyuwenyu): must load_tuning_state before ema instance building
        if self.cfg.tuning:
            print(f'tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.model = dist_utils.warp_model(self.model.to(device), sync_bn=cfg.sync_bn, \
            find_unused_parameters=cfg.find_unused_parameters)

        self.criterion = to(cfg.criterion, device)
        self.postprocessor = to(cfg.postprocessor, device)
        self.teacher_model = self._build_teacher_model(device)
        if self.teacher_model is not None and hasattr(self.criterion, 'teacher_model'):
            self.criterion.teacher_model = self.teacher_model

        self.ema = to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch
        
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text(f'config', '{:s}'.format(cfg.__repr__()), 0)

    def _build_teacher_model(self, device):
        yaml_cfg = getattr(self.cfg, 'yaml_cfg', {}) or {}
        teacher_config = yaml_cfg.get('teacher_config')
        teacher_checkpoint = yaml_cfg.get('teacher_checkpoint')
        if not teacher_config or not teacher_checkpoint:
            return None
        teacher_cfg = YAMLConfig(str(teacher_config), device=str(device), resume=None)
        teacher = teacher_cfg.model
        state = torch.load(str(teacher_checkpoint), map_location='cpu')
        if isinstance(state, dict) and isinstance(state.get('ema'), dict) and 'module' in state['ema']:
            params = state['ema']['module']
            weight_object = 'ema.module'
        elif isinstance(state, dict) and 'model' in state:
            params = state['model']
            weight_object = 'model'
        else:
            params = state
            weight_object = 'raw'
        missing, unexpected = teacher.load_state_dict(params, strict=False)
        teacher = teacher.to(device).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        print(
            f"Loaded teacher_model from {teacher_checkpoint} ({weight_object}); "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
        return teacher

    def _apply_trainable_rules(self):
        yaml_cfg = getattr(self.cfg, 'yaml_cfg', {}) or {}
        include = yaml_cfg.get('trainable_param_patterns') or []
        exclude = yaml_cfg.get('frozen_param_patterns') or []
        if not include and not exclude:
            return
        include_re = [re.compile(pattern) for pattern in include]
        exclude_re = [re.compile(pattern) for pattern in exclude]
        total = trainable = 0
        trainable_names = []
        for name, param in dist_utils.de_parallel(self.model).named_parameters():
            total += param.numel()
            keep = param.requires_grad
            if include_re:
                keep = any(pattern.search(name) for pattern in include_re)
            if keep and exclude_re:
                keep = not any(pattern.search(name) for pattern in exclude_re)
            param.requires_grad_(keep)
            if keep:
                trainable += param.numel()
                trainable_names.append(name)
        print(
            f"Applied trainable rules: trainable_params={trainable} / total_params={total}; "
            f"trainable_tensors={len(trainable_names)}",
            flush=True,
        )
        preview = ', '.join(trainable_names[:20])
        if preview:
            suffix = ' ...' if len(trainable_names) > 20 else ''
            print(f"Trainable preview: {preview}{suffix}", flush=True)

    def cleanup(self, ):
        if self.writer:
            atexit.register(self.writer.close)

    def train(self, ):
        self._setup()
        self._apply_trainable_rules()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        disable_eval = bool(getattr(self.cfg, 'disable_eval', False))
        if disable_eval:
            self.val_dataloader = None
            self.evaluator = None
            self.test_dataloader = None
            self.test_evaluator = None
        else:
            self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
                shuffle=self.cfg.val_dataloader.shuffle)

        set_total_updates = getattr(self.criterion, 'set_total_updates', None)
        if callable(set_total_updates):
            set_total_updates(len(self.train_dataloader) * int(self.cfg.epoches))

        if not disable_eval:
            self.evaluator = self.cfg.evaluator

        evaluate_test_during_training = bool(
            getattr(self.cfg, 'yaml_cfg', {}).get('evaluate_test_during_training', True)
        )
        if not disable_eval and evaluate_test_during_training and self.cfg.test_dataloader is not None:
            self.test_dataloader = dist_utils.warp_loader(self.cfg.test_dataloader, \
                shuffle=self.cfg.test_dataloader.shuffle)
            self.test_evaluator = self.cfg.test_evaluator
        elif not disable_eval:
            self.test_dataloader = None
            self.test_evaluator = None

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self, ):
        self._setup()

        if self.cfg.test_dataloader is not None:
            self.val_dataloader = dist_utils.warp_loader(self.cfg.test_dataloader, \
                shuffle=self.cfg.test_dataloader.shuffle)
            self.evaluator = self.cfg.test_evaluator
            print('Using test_dataloader for --test-only evaluation')
        else:
            self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader, \
                shuffle=self.cfg.val_dataloader.shuffle)
            self.evaluator = self.cfg.evaluator
        
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def to(self, device):
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                v.to(device)

    def state_dict(self):
        """state dict, train/eval
        """
        state = {}
        state['date'] = datetime.now().isoformat()
        
        # TODO for resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict() 

        return state


    def load_state_dict(self, state):
        """load state dict, train/eval
        """
        # TODO
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                print(f'Not load {k}.state_dict')


    def load_resume_state(self, path: str):
        """load resume
        """
        # for cuda:0 memory
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        self.load_state_dict(state)

    
    def load_tuning_state(self, path: str,):
        """only load model for tuning and skip missed/dismatched keys
        """
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist_utils.de_parallel(self.model)
        
        # TODO hard code
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')


    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}


    def fit(self, ):
        raise NotImplementedError('')


    def val(self, ):
        raise NotImplementedError('')
