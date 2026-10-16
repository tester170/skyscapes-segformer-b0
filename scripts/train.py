"""Train or fine-tune SegFormer-B0 on SkyScapes-Dense with the MMEngine Runner (same pipeline as the experiments).

  --init best     fine-tune from the bundled trained model (checkpoints/segformer_b0_skyscapes_ep220.pth)
  --init encoder  train from scratch with the LoveDA-pretrained MiT-B0 encoder (as the original experiment)
  --init PATH     start from any MMSeg checkpoint (.pth)
If <work-dir>/last_checkpoint exists, training resumes from it (e.g. after a Colab disconnect).
Usage: python scripts/train.py --cache /content/cache --work-dir <dir> [--init best] [--max-epochs 600] [...]"""
import argparse
import sys
from pathlib import Path

import common

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--work-dir', required=True)
ap.add_argument('--init', default='best')
ap.add_argument('--max-epochs', type=int, default=600, help='upper bound; early stopping usually ends sooner')
ap.add_argument('--val-interval', type=int, default=10, help='validation every N epochs')
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--workers', type=int, default=6, help='dataloader workers (capped by the number of CPUs)')
ap.add_argument('--warmup-iters', type=int, default=1500)
ap.add_argument('--fp32', action='store_true', help='disable fp16 mixed precision')
ap.add_argument('--progress-interval', type=float, default=30.0, help='seconds between progress blocks')
a = ap.parse_args()

common.cap_vram()
cfg = common.load_cfg(a.cache)
wd = Path(a.work_dir)
wd.mkdir(parents=True, exist_ok=True)
cfg.work_dir = str(wd)
cfg.train_cfg.max_epochs = a.max_epochs
cfg.train_cfg.val_interval = a.val_interval
cfg.train_dataloader.batch_size = a.batch
cfg.train_dataloader.num_workers = max(1, min(a.workers, common.cpu_count()))
cfg.param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=a.warmup_iters)]
for h in cfg.custom_hooks:
    if h['type'] == 'PlateauControlHook':
        h['start_iter'] = a.warmup_iters
    if h['type'] == 'ConsoleProgressHook':
        h['interval_s'] = a.progress_interval
if not a.fp32:
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.dtype = 'float16'
    cfg.optim_wrapper.loss_scale = 'dynamic'
init = {'best': common.CKPT_BEST, 'encoder': common.CKPT_ENCODER}.get(a.init, a.init)
for key, path in (('encoder', common.CKPT_ENCODER), ('best', common.CKPT_BEST)):
    if a.init == key and not path.exists():      # GitHub release; the encoder can also be rebuilt from Hugging Face
        import subprocess
        what = 'model' if key == 'best' else 'encoder'
        if subprocess.run([sys.executable, str(common.BUNDLE / 'scripts' / 'fetch.py'), what]).returncode and key == 'encoder':
            subprocess.run([sys.executable, str(common.BUNDLE / 'scripts' / 'get_encoder.py')], check=True)
        if not path.exists():
            sys.exit(f'{path} not found')
if (wd / 'last_checkpoint').exists():
    cfg.resume, cfg.load_from = True, None      # mmengine finds the latest checkpoint in work_dir
    print(f'resuming from {(wd / "last_checkpoint").read_text().strip()}', flush=True)
else:
    cfg.resume, cfg.load_from = False, str(init)
    print(f'initial weights: {init}', flush=True)
print(f'batch {cfg.train_dataloader.batch_size}, workers {cfg.train_dataloader.num_workers}, '
      f'max epochs {a.max_epochs}, validation every {a.val_interval} epochs, fp16 {not a.fp32}', flush=True)

from mmengine.runner import Runner  # noqa: E402
Runner.from_cfg(cfg).train()
best = sorted(wd.glob('best_mIoU_epoch_*.pth'))
print('best checkpoint:', best[-1] if best else 'none (no validation yet)')
