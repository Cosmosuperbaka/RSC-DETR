# Modified for the RSC-DETR final source package; see NOTICE and docs/FINAL_VERSION_AUDIT.json.
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
import os
import shlex
import signal
import sys

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):

    def _evaluate_on_configured_device(self, module, evaluator, dataloader, epoch, label):
        eval_device_name = getattr(self.cfg, 'eval_device', self.cfg.yaml_cfg.get('eval_device', None))
        eval_device = torch.device(eval_device_name) if eval_device_name else self.device
        train_device = self.device

        if eval_device != train_device:
            print(f'--- moving {label} evaluation module to {eval_device} (epoch {epoch}) ---')
            module.to(eval_device)
            self.criterion.to(eval_device)
            self.postprocessor.to(eval_device)

        try:
            return evaluate(
                module,
                self.criterion,
                self.postprocessor,
                dataloader,
                evaluator,
                eval_device,
                max_eval_batches=getattr(self.cfg, 'max_eval_batches', None)
            )
        finally:
            if eval_device != train_device:
                print(f'--- moving {label} evaluation module back to {train_device} (epoch {epoch}) ---')
                module.to(train_device)
                self.criterion.to(train_device)
                self.postprocessor.to(train_device)
                if torch.cuda.is_available() and train_device.type == 'cuda':
                    torch.cuda.empty_cache()

    def _best_state_dict(self):
        if not getattr(self.cfg, 'save_best_model_only', False):
            return self.state_dict()

        state = {
            'date': datetime.datetime.now().isoformat(),
            'last_epoch': self.last_epoch,
            'checkpoint_type': 'model_only_best',
            'model': dist_utils.de_parallel(self.model).state_dict(),
        }
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        return state

    def _criterion_diagnostics(self):
        """Expose scalar training-only coordinator diagnostics in epoch logs."""
        criterion = dist_utils.de_parallel(self.criterion)
        diagnostics = getattr(criterion, 'last_x1_diagnostics', {}) or {}
        result = {}
        for name, value in diagnostics.items():
            if hasattr(value, 'detach'):
                value = value.detach().float()
                if value.numel() != 1:
                    value = value.mean()
                result[f'coord_{name}'] = value.item()
            elif isinstance(value, (int, float)):
                result[f'coord_{name}'] = float(value)
        balance = getattr(criterion, 'balance', None)
        if balance is not None:
            for name in ('lambda_position', 'x3_primary_score', 'x3_static_gate', 'x3_class_static_gate', 'x3_loc_static_gate', 'x1_loc_static_gate', 'x1_aux_x3_static_gate'):
                value = getattr(balance, name, None)
                if value is not None:
                    result[f'coord_{name}'] = float(value)
        return result

    def _fusion_diagnostics(self):
        module = dist_utils.de_parallel(self.ema.module if self.ema else self.model)
        fusion = getattr(module, 'fusion', None)
        if fusion is None:
            return {}
        layers = getattr(fusion, 'fusions', [])
        stats = {}
        sem_spsf = getattr(module, 'sem_spsf', None)
        for name, value in (getattr(sem_spsf, 'last_stats', {}) or {}).items():
            if isinstance(value, bool):
                stats[f'sem_spsf_{name}'] = float(value)
            elif isinstance(value, (int, float)):
                stats[f'sem_spsf_{name}'] = float(value)
        decoder = getattr(module, 'decoder', None)
        for name, value in (getattr(decoder, 'last_ca_spsf_debug', {}) or {}).items():
            if isinstance(value, (int, float)):
                stats[f'ca_spsf_{name}'] = float(value)
        for idx, layer in enumerate(layers):
            prefix = f'fusion_s{idx + 3}'
            for name in [
                'last_common_norm',
                'last_private_norm',
                'last_weighted_private_norm',
                'last_alpha',
                'last_pr_rms',
                'last_nr_rms',
                'last_residual_norm',
                'last_gain_mean',
                'last_gain_std',
                'last_drop_ratio',
                'last_private_sparsity',
                'last_private_spatial_cv',
                'last_effective_alpha',
                'last_base_norm',
                'last_candidate_norm',
                'last_delta_norm',
                'last_output_delta_norm',
                'last_delta_base_ratio',
                'last_output_delta_base_ratio',
                'last_output_norm',
                'last_gate_mean',
                'last_gate_std',
                'last_risk',
                'last_semantic_risk',
                'last_energy_risk',
                'last_private_risk',
                'last_strength',
                'last_progress',
                'last_semantic_cos',
                'last_private_common_ratio',
            ]:
                value = getattr(layer, name, None)
                if value is None:
                    continue
                if hasattr(value, 'detach'):
                    value = value.detach().float().mean().item()
                else:
                    value = float(value)
                stats[f'{prefix}_{name}'] = value
            alpha_logit = getattr(layer, 'alpha_logit', None)
            if alpha_logit is not None:
                stats[f'{prefix}_alpha_logit'] = alpha_logit.detach().float().mean().item()
                stats[f'{prefix}_alpha'] = alpha_logit.detach().float().sigmoid().mean().item()
            alpha = getattr(layer, 'alpha', None)
            if alpha is not None:
                stats[f'{prefix}_residual_alpha_param'] = alpha.detach().float().mean().item()
            rms = getattr(layer, 'shared_rmsnorm', None)
            if rms is not None and hasattr(rms, 'gain'):
                stats[f'{prefix}_rmsnorm_gain_mean'] = rms.gain.detach().float().mean().item()
                stats[f'{prefix}_rmsnorm_gain_std'] = rms.gain.detach().float().std().item()
            theta = getattr(layer, 'theta', None)
            if theta is not None and hasattr(layer, 'channel_alpha'):
                alpha = layer.channel_alpha().detach().float().flatten()
                if alpha.numel() > 0:
                    q = torch.quantile(alpha, alpha.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
                    stats[f'{prefix}_theta_mean'] = theta.detach().float().mean().item()
                    stats[f'{prefix}_theta_std'] = theta.detach().float().std(unbiased=False).item()
                    stats[f'{prefix}_alpha_mean'] = alpha.mean().item()
                    stats[f'{prefix}_alpha_std'] = alpha.std(unbiased=False).item()
                    stats[f'{prefix}_alpha_min'] = alpha.min().item()
                    stats[f'{prefix}_alpha_max'] = alpha.max().item()
                    stats[f'{prefix}_alpha_p10'] = q[0].item()
                    stats[f'{prefix}_alpha_p25'] = q[1].item()
                    stats[f'{prefix}_alpha_p50'] = q[2].item()
                    stats[f'{prefix}_alpha_p75'] = q[3].item()
                    stats[f'{prefix}_alpha_p90'] = q[4].item()
                    stats[f'{prefix}_alpha_lt010_ratio'] = (alpha < 0.10).float().mean().item()
                    stats[f'{prefix}_alpha_lt025_ratio'] = (alpha < 0.25).float().mean().item()
                    stats[f'{prefix}_alpha_045_055_ratio'] = ((alpha >= 0.45) & (alpha <= 0.55)).float().mean().item()
                    stats[f'{prefix}_alpha_gt075_ratio'] = (alpha > 0.75).float().mean().item()
                    stats[f'{prefix}_alpha_gt090_ratio'] = (alpha > 0.90).float().mean().item()
                    stats[f'{prefix}_theta_grad_norm'] = float(getattr(layer, '_csspsf_last_theta_grad_norm', 0.0))
        return stats

    def _restore_ap_best_from_log(self):
        """Restore the primary COCO-AP best state when resuming training.

        ``best_stat`` is not part of a training checkpoint.  Reconstructing it
        from the current run's contiguous log suffix prevents a resumed job
        from forgetting a better AP obtained before the resume point.
        """
        best_stat = {'epoch': -1}
        if self.last_epoch < 0 or not self.output_dir:
            return best_stat
        log_path = self.output_dir / 'log.txt'
        if not log_path.exists():
            return best_stat
        min_save_epoch = int(getattr(self.cfg, 'min_save_epoch', self.cfg.yaml_cfg.get('min_save_epoch', -1)))

        current_run = []
        with log_path.open('r') as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                epoch = record.get('epoch')
                if not isinstance(epoch, int):
                    continue
                if current_run and epoch <= current_run[-1].get('epoch', -1):
                    current_run = []
                current_run.append(record)

        for record in current_run:
            epoch = record['epoch']
            if epoch < min_save_epoch:
                continue
            if epoch > self.last_epoch:
                continue
            for field, values in record.items():
                if not field.endswith('coco_eval_bbox') or not isinstance(values, list) or not values:
                    continue
                key = 'coco_eval_bbox'
                current_ap = values[0]
                if key not in best_stat or current_ap > best_stat[key]:
                    best_stat['epoch'] = epoch
                    best_stat[key] = current_ap
                    best_stat[f'{key}_ap50'] = values[1] if len(values) > 1 else 0.0
                    best_stat[f'{key}_ap75'] = values[2] if len(values) > 2 else 0.0
        if best_stat['epoch'] >= 0:
            print(f"Restored AP-best state from log: {best_stat}")
        return best_stat
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        def _save_interrupt_and_exit(signum, _frame):
            if self.output_dir and dist_utils.is_main_process():
                interrupt_path = self.output_dir / 'interrupt.pth'
                state = self.state_dict()
                dist_utils.save_on_master(state, interrupt_path)
                if getattr(self.cfg, 'save_last_on_interrupt', True):
                    last_path = self.output_dir / 'last.pth'
                    try:
                        if last_path.exists():
                            last_path.unlink()
                        os.link(interrupt_path, last_path)
                    except OSError:
                        dist_utils.save_on_master(state, last_path)
                (self.output_dir / 'interrupted.flag').write_text(
                    f'收到信号 {signum}，已保存 interrupt.pth，时间 {datetime.datetime.now().isoformat()}\\n',
                    encoding='utf-8',
                )
                argv = [sys.executable] + sys.argv
                resume_cmd = ' '.join(shlex.quote(str(item)) for item in argv)
                if '--resume' not in sys.argv and '-r' not in sys.argv:
                    resume_cmd += ' --resume ' + shlex.quote(str(interrupt_path))
                (self.output_dir / 'resume_command.sh').write_text(
                    '#!/usr/bin/env bash\\nset -e\\ncd ' + shlex.quote(os.getcwd()) + '\\n'
                    + resume_cmd + '\\n',
                    encoding='utf-8',
                )
                os.chmod(self.output_dir / 'resume_command.sh', 0o755)
                print(f'Interrupted by signal {signum}; saved {interrupt_path}')
            raise SystemExit(128 + int(signum))

        previous_sigint = signal.getsignal(signal.SIGINT)
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _save_interrupt_and_exit)
        signal.signal(signal.SIGTERM, _save_interrupt_and_exit)

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        # COCO AP@[0.50:0.95] is the sole primary checkpoint-selection metric.
        # AP50/AP75 are reported from the same epoch but never participate in
        # selecting the checkpoint.
        disable_eval = bool(getattr(args, 'disable_eval', False))
        if disable_eval:
            best_stat = {'epoch': -1, 'disabled': True}
            print('Epoch-end validation/test evaluation is disabled; best.pth will not be selected by AP.')
        else:
            best_stat = self._restore_ap_best_from_log()

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):
            train_module = dist_utils.de_parallel(self.model)
            fusion = getattr(train_module, 'fusion', None)
            if fusion is not None and hasattr(fusion, 'set_epoch'):
                fusion.set_epoch(epoch)
            if self.ema is not None and getattr(self.ema, 'module', None) is not None:
                ema_fusion = getattr(dist_utils.de_parallel(self.ema.module), 'fusion', None)
                if ema_fusion is not None and hasattr(ema_fusion, 'set_epoch'):
                    ema_fusion.set_epoch(epoch)

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                max_train_batches=getattr(args, 'max_train_batches', None),
                gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', args.yaml_cfg.get('gradient_accumulation_steps', 1)),
                runtime_diagnostics=getattr(args, 'runtime_diagnostics', args.yaml_cfg.get('runtime_diagnostics', False)),
                runtime_diagnostics_interval=getattr(args, 'runtime_diagnostics_interval', args.yaml_cfg.get('runtime_diagnostics_interval', 200)),
                runtime_diagnostics_path=getattr(args, 'runtime_diagnostics_path', args.yaml_cfg.get('runtime_diagnostics_path', '')),
                online_probe_path=getattr(args, 'online_probe_path', args.yaml_cfg.get('online_probe_path', '')),
                online_probe_interval=getattr(args, 'online_probe_interval', args.yaml_cfg.get('online_probe_interval', 1))
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1
            min_save_epoch = int(getattr(args, 'min_save_epoch', args.yaml_cfg.get('min_save_epoch', -1)))
            allow_checkpoint_save = epoch >= min_save_epoch

            if self.output_dir:
                checkpoint_paths = []
                if allow_checkpoint_save and getattr(args, 'save_last', True) and getattr(args, 'save_last_every_epoch', True):
                    checkpoint_paths.append(self.output_dir / 'last.pth')
                # extra checkpoint before LR drop and every 100 epochs
                if allow_checkpoint_save and getattr(args, 'save_checkpoints', True) and (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            if disable_eval:
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **self._fusion_diagnostics(),
                    **self._criterion_diagnostics(),
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
                if self.output_dir and dist_utils.is_main_process():
                    with (self.output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                stop_after_epoch = getattr(args, 'stop_after_epoch', None)
                if stop_after_epoch is not None and epoch >= int(stop_after_epoch):
                    print(f'Stopping early after epoch {epoch} due to stop_after_epoch={stop_after_epoch}; original epoches={args.epoches}')
                    break
                continue

            module = self.ema.module if self.ema else self.model
            select_best_by_test = bool(getattr(args, 'select_best_by_test', False))
            skip_val_eval = bool(getattr(args, 'skip_val_eval', False))
            eval_dataloader = self.test_dataloader if select_best_by_test and self.test_dataloader is not None else self.val_dataloader
            eval_evaluator = self.test_evaluator if select_best_by_test and self.test_evaluator is not None else self.evaluator
            eval_name = 'test' if eval_dataloader is self.test_dataloader else 'val'

            if skip_val_eval and eval_name == 'val' and self.test_dataloader is not None:
                eval_dataloader = self.test_dataloader
                eval_evaluator = self.test_evaluator
                eval_name = 'test'

            print(f'--- {eval_name} set evaluation for best selection (epoch {epoch}) ---')
            test_stats, coco_evaluator = self._evaluate_on_configured_device(
                module, eval_evaluator, eval_dataloader, epoch, eval_name)

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'{eval_name}/{k}_{i}'.format(k), v, epoch)

                # Official RT-DETR/COCO convention: select solely by the
                # primary AP@[0.50:0.95] metric.
                cur_ap95 = test_stats[k][0]
                cur_ap50 = test_stats[k][1] if len(test_stats[k]) > 1 else 0.0
                cur_ap75 = test_stats[k][2] if len(test_stats[k]) > 2 else 0.0
                improved = k not in best_stat or cur_ap95 > best_stat[k]
                if improved:
                    best_stat['epoch'] = epoch
                    best_stat[k] = cur_ap95
                    best_stat[f'{k}_ap50'] = cur_ap50
                    best_stat[f'{k}_ap75'] = cur_ap75

                if improved and allow_checkpoint_save and self.output_dir and getattr(args, 'save_best', True):
                    dist_utils.save_on_master(self._best_state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            test_log_stats = {}
            if self.test_dataloader is not None and eval_dataloader is not self.test_dataloader:
                print(f'--- Test set evaluation (epoch {epoch}) ---')
                test_stats_test, _ = self._evaluate_on_configured_device(
                    module, self.test_evaluator, self.test_dataloader, epoch, 'test')
                print(f'test_set_stats: {test_stats_test}')
                test_log_stats = {f'test_{k}': v for k, v in test_stats_test.items()}

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'{eval_name}_{k}': v for k, v in test_stats.items()},
                **test_log_stats,
                **self._fusion_diagnostics(),
                **self._criterion_diagnostics(),
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            stop_after_epoch = getattr(args, 'stop_after_epoch', None)
            if stop_after_epoch is not None and epoch >= int(stop_after_epoch):
                print(f'Stopping early after epoch {epoch} due to stop_after_epoch={stop_after_epoch}; original epoches={args.epoches}')
                break

        if self.output_dir and getattr(args, 'save_last', True):
            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'last.pth')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device,
                max_eval_batches=getattr(self.cfg, 'max_eval_batches', None))
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
