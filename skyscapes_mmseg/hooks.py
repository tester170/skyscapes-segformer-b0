"""Training-control hooks.

VramLogHook          logs peak reserved memory (the allocator cache WDDM must hold, not only allocated tensors).
PlateauControlHook   for iteration-based training (mmengine's ReduceOnPlateau supports only by_epoch=True):
                     after every validation (i) saves the per-frame confusion matrices to
                     work_dir/val_confusion/iter_<n>.json, (ii) multiplies all LRs by `factor` when `monitor` has not
                     improved by `min_delta` for `lr_patience` validations (then waits `cooldown` validations),
                     (iii) stops training after `stop_patience` validations without improvement.
                     Validations before `start_iter` (warm-up) are not counted. Its state is stored in checkpoints,
                     so a resumed run continues the same counters; the LRs themselves are restored with the optimizer.
Best checkpoints are saved by mmengine CheckpointHook(save_best=...).
"""
import json
import os
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class VramLogHook(Hook):
    priority = 'LOW'

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        runner.message_hub.update_scalar('train/reserved_mb', torch.cuda.max_memory_reserved() / 2 ** 20)


@HOOKS.register_module()
class PlateauControlHook(Hook):
    """LR reduction on plateau + early stopping for any train loop, decided on an EMA of the monitored metric
    (ema = alpha * value + (1 - alpha) * ema), so single noisy validations on 2 frames do not trigger actions.
    Improvement = EMA above its best by more than min_delta. After lr_patience validations without improvement:
    LR x factor, then `cooldown` validations are not counted towards the next reduction; at most max_reductions.
    After stop_patience validations without improvement: stop. Validations before start_iter (warm-up) only feed the
    EMA. Also saves val (per frame) and train-crop confusion matrices of every validation to work_dir/val_confusion.
    State is stored in checkpoints for resume."""
    priority = 'LOW'  # after CheckpointHook (best checkpoints on the raw metric), before ConsoleProgressHook

    def __init__(self, monitor='mIoU', ema_alpha=0.5, min_delta=0.1, lr_patience=5, factor=0.5, cooldown=2,
                 max_reductions=4, stop_patience=12, start_iter=0):
        self.monitor, self.ema_alpha, self.min_delta = monitor, ema_alpha, min_delta
        self.lr_patience, self.factor, self.cooldown, self.max_reductions = lr_patience, factor, cooldown, max_reductions
        self.stop_patience, self.start_iter = stop_patience, start_iter
        self.state = dict(ema=None, best=None, best_iter=None, best_raw=None, best_raw_iter=None, wait=0,
                          since_reduce=0, cooldown_left=0, reductions=0, history=[])

    def _save_confusion(self, runner, metrics):
        from .patches import TRAIN_CONFUSION as tc
        frames = runner.val_evaluator.metrics[0].last_frames
        out = Path(runner.work_dir) / 'val_confusion'
        out.mkdir(exist_ok=True)
        path = out / f'iter_{runner.iter}.json'
        tmp = path.with_suffix('.tmp')
        train_cm = None if tc.cm is None else tc.cm.reshape(tc.k, tc.k).cpu().numpy().tolist()
        metric = runner.val_evaluator.metrics[0]
        levels_cm = None
        if metric.last_level_cms:
            from .seg_metrics import level_key
            levels_cm = {f'{c}{n}': metric.last_level_cms[level_key(G)].tolist()
                         for c, steps in metric.chains.items() for n, G, _ in steps[1:]}
        tmp.write_text(json.dumps(dict(iter=runner.iter, epoch=runner.epoch,
                                       metrics={k: float(v) for k, v in metrics.items()},
                                       lr=[g['lr'] for g in runner.optim_wrapper.optimizer.param_groups][:1],
                                       frames={s: c.tolist() for s, c in frames.items()},
                                       train_crops_cm=train_cm, levels_cm=levels_cm)))
        os.replace(tmp, path)

    def after_val_epoch(self, runner, metrics=None):
        if not metrics or self.monitor not in metrics:
            return
        self._save_confusion(runner, metrics)
        s, raw, it = self.state, float(metrics[self.monitor]), runner.iter
        s['ema'] = raw if s['ema'] is None else self.ema_alpha * raw + (1 - self.ema_alpha) * s['ema']
        s['history'].append((it, raw, s['ema']))
        if s['best_raw'] is None or raw > s['best_raw']:
            s.update(best_raw=raw, best_raw_iter=it)
        if it < self.start_iter:
            return
        log = runner.logger.info
        if s['best'] is None or s['ema'] > s['best'] + self.min_delta:
            s.update(best=s['ema'], best_iter=it, wait=0, since_reduce=0)
            log(f'[plateau] {self.monitor} EMA improved to {s["ema"]:.3f} (raw {raw:.3f}) at iter {it}')
            return
        s['wait'] += 1
        if s['cooldown_left'] > 0:
            s['cooldown_left'] -= 1
        else:
            s['since_reduce'] += 1
        log(f'[plateau] no improvement {s["wait"]}/{self.stop_patience}: EMA {s["ema"]:.3f} vs best {s["best"]:.3f} '
            f'(+{self.min_delta} needed), raw {raw:.3f}')
        if s['since_reduce'] >= self.lr_patience and s['reductions'] < self.max_reductions:
            for g in runner.optim_wrapper.optimizer.param_groups:
                g['lr'] *= self.factor
            s.update(since_reduce=0, cooldown_left=self.cooldown, reductions=s['reductions'] + 1)
            log(f'[plateau] LR x{self.factor} (reduction {s["reductions"]}/{self.max_reductions}), '
                f'base lr now {runner.optim_wrapper.optimizer.param_groups[0]["lr"]:.3e}')
        if s['wait'] >= self.stop_patience:
            runner.train_loop.stop_training = True
            log(f'[plateau] early stop at iter {it}: best EMA {s["best"]:.3f} at {s["best_iter"]}, '
                f'best raw {s["best_raw"]:.3f} at {s["best_raw_iter"]}')

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {})['plateau_control'] = dict(self.state)

    def after_load_checkpoint(self, runner, checkpoint):
        state = checkpoint.get('meta', {}).get('plateau_control')
        if state:
            self.state.update(state)


@HOOKS.register_module()
class EpochEquivalentHook(Hook):
    """Logs train/epoch_eq = crops seen / crops needed to cover the training area once (crop area units)."""
    priority = 'LOW'

    def before_train(self, runner):
        ds = runner.train_dataloader.dataset
        area = sum(h * w for h, w in ds.shapes)
        self.per_iter = runner.train_dataloader.batch_size * ds.crop_size[0] * ds.crop_size[1] / area
        runner.logger.info(f'[epoch-eq] 1 epoch equivalent = {1 / self.per_iter:.1f} iterations')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        runner.message_hub.update_scalar('train/epoch_eq', (runner.iter + 1) * self.per_iter)


def _hms(sec):
    sec = int(max(sec, 0))
    return f'{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


@HOOKS.register_module()
class ConsoleProgressHook(Hook):
    """Human-readable status block in the job console every `interval_s` seconds and after every validation:
    progress bar, s/it, elapsed / remaining to the iteration cap, epoch equivalents, losses, LR, VRAM, training-crop
    metrics since the last validation (confusion accumulated on the GPU inside the head's accuracy call), the last
    validation (20 classes, L1..L5, chain curves, K60, kappa), best monitored value and plateau counters."""
    priority = 'VERY_LOW'

    def __init__(self, name='', interval_s=20.0, bar=30):
        self.name, self.interval_s, self.bar = name, interval_s, bar

    def before_train(self, runner):
        import time
        from .patches import TRAIN_CONFUSION
        model = getattr(runner.model, 'module', runner.model)
        head = getattr(model, 'decode_head', None)
        if head is not None:
            head.track_train_confusion = True
        TRAIN_CONFUSION.reset()
        self.tc = TRAIN_CONFUSION
        self.name = self.name or Path(runner.work_dir).name
        self.t_last = self.t_start = time.time()
        self.it_start = runner.iter
        self.times = []
        self.train_flat = None
        self.it_per_ep = len(runner.train_dataloader)
        self.plateau = next((h for h in runner.hooks if isinstance(h, PlateauControlHook)), None)
        metric = runner.val_evaluator.metrics[0]
        self.chains, self.levels = metric.chains, metric.levels

    def _train_metrics(self, reset=False):
        from .seg_metrics import evaluate_all
        if self.tc.cm is None:
            return
        cm = self.tc.cm.reshape(self.tc.k, self.tc.k).cpu().numpy()
        if reset:
            self.tc.reset()
        self.train_flat = evaluate_all(cm, self.chains, self.levels)[0]

    def _block(self, runner, outputs=None, after_val=False):
        import time
        it, mx = runner.iter + (0 if after_val else 1), runner.train_loop.max_iters
        now = time.time()
        s_it = (sum(self.times[-200:]) / len(self.times[-200:])) if self.times else float('nan')
        done = int(self.bar * it / mx)
        lr = runner.optim_wrapper.optimizer.param_groups[0]['lr']
        mep = getattr(runner.train_loop, 'max_epochs', None)
        ep_txt = f'эпоха {it / self.it_per_ep:.1f}/{mep}' if mep else f'эпох {it / self.it_per_ep:.1f}'
        lines = [f'━━━ {self.name} ' + '━' * 40,
                 f'[{"█" * done}{"░" * (self.bar - done)}] {ep_txt} | итерация {it}/{mx} ({100 * it / mx:5.1f}%) | '
                 f'{s_it:.3f} с/итер | прошло {_hms(now - self.t_start)} | до лимита ≤{_hms((mx - it) * s_it)}']
        if outputs:
            losses = ' '.join(f'{k.replace("decode.", "")} {float(v):.4f}' for k, v in outputs.items()
                              if 'loss' in k and k.count('.') <= 1 and not k.startswith('decode.d'))
            lines.append(f'потери: {losses} | lr {lr:.2e} | VRAM резерв {torch.cuda.max_memory_reserved() / 2 ** 30:.1f} ГБ')
        tf = self.train_flat
        if tf:
            lv = ' '.join(f'{al} {tf[f"mIoU_{al}"]:.2f}' for al in self.levels if f'mIoU_{al}' in tf)
            lines.append(f'train (патчи эпох с последней валидации, сетка + 25 % редких классов): mIoU20 {tf["mIoU"]:.2f}'
                         + (f' | {lv}' if lv else '') + f' | kappa {tf["kappa"] / 100:.4f} | OA {tf["aAcc"]:.2f}')
        vf = runner.val_evaluator.metrics[0].last_flat
        if vf:
            lines.append((f'val loss {vf["val_loss"]:.4f} | ' if 'val_loss' in vf else '') + f'val: mIoU20 {vf["mIoU"]:.2f}' + ''.join(f' {a} {vf[f"mIoU_{a}"]:.2f}' for a in self.levels)
                         + f' | kappa {vf["kappa"] / 100:.4f} | mDice {vf["mDice"]:.2f} | OA {vf["aAcc"]:.2f} | '
                         + ' '.join(f'K60_{c} {int(vf[f"K60_{c}"])}' for c in self.chains))
            for c, steps in self.chains.items():
                lines.append(f'  цепочка {c} (классов:mIoU) ' + ' '.join(f'{n}:{vf[f"mIoU_{c}{n}"]:.1f}' for n, _, _ in steps))
        p = self.plateau
        if p and p.state['best'] is not None:
            s, vi = p.state, runner.train_loop.val_interval
            unit = 'эпох' if mep else 'итер.'
            lines.append(f'{p.monitor}: сейчас {s["history"][-1][1]:.2f} (EMA {s["ema"]:.2f}) | лучшее EMA {s["best"]:.2f} '
                         f'@ итер. {s["best_iter"]} | лучшее сырое {s["best_raw"]:.2f} @ {s["best_raw_iter"]}')
            lines.append(f'без улучшения {s["wait"]}/{p.stop_patience} валидаций ({s["wait"] * vi} {unit}) | '
                         f'снижение LR x{p.factor} через {max(p.lr_patience - s["since_reduce"], 0) + s["cooldown_left"]} вал. | '
                         f'снижений {s["reductions"]}/{p.max_reductions} | ранняя остановка не позже чем через '
                         f'{(p.stop_patience - s["wait"]) * vi} {unit}')
        print('\n'.join(lines), flush=True)
        self.t_last = now

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        import time
        now = time.time()
        self.times.append(runner.message_hub.get_scalar('train/time').current()
                          if 'train/time' in runner.message_hub.log_scalars else now - self.t_last)
        if now - self.t_last >= self.interval_s:
            self._train_metrics()
            self._block(runner, outputs)

    def after_val_epoch(self, runner, metrics=None):
        self._train_metrics(reset=True)
        self._block(runner, after_val=True)
