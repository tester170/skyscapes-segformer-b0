"""Evaluate a checkpoint on the validation frames exactly as during training (slide 1024/768 at full resolution,
fp32), save all metrics and confusion matrices (20 classes and every merge level, probability-sum merging) to JSON
and draw the confusion-matrix figures (200 dpi, per-class kappa on the diagonal).
Usage: python scripts/evaluate.py --cache /content/cache --checkpoint <ckpt.pth> --out <dir>"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import common

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--checkpoint', default=str(common.CKPT_BEST))
ap.add_argument('--out', required=True)
ap.add_argument('--title', default='SegFormer-B0')
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

common.cap_vram()
cfg = common.load_cfg(a.cache)
cfg.load_from = a.checkpoint
cfg.work_dir = str(out / 'eval_workdir')
cfg.custom_hooks = []
from mmengine.runner import Runner  # noqa: E402
from skyscapes_mmseg.seg_metrics import level_key  # noqa: E402

runner = Runner.from_cfg(cfg)
metrics = runner.test()
m = runner.test_evaluator.metrics[0]
levels_cm = {f'{c}{n}': m.last_level_cms[level_key(G)].tolist()
             for c, steps in m.chains.items() for n, G, _ in steps[1:]} if m.last_level_cms else None
res = dict(checkpoint=a.checkpoint, metrics={k: float(v) for k, v in metrics.items()},
           flat={k: float(v) for k, v in m.last_flat.items()},
           frames={s: c.tolist() for s, c in m.last_frames.items()}, levels_cm=levels_cm)
js = out / 'eval.json'
tmp = js.with_suffix('.tmp')
tmp.write_text(json.dumps(res))
os.replace(tmp, js)
print({k: round(v, 3) for k, v in res['metrics'].items() if k in ('mIoU', 'kappa', 'aAcc', 'mDice', 'val_loss')})
subprocess.run([sys.executable, str(common.BUNDLE / 'scripts' / 'plot_confusion.py'), str(js), str(out / 'figures'),
                '--merges', str(common.MERGES), '--chain-title', 'none', '--title-prefix', a.title], check=True)
