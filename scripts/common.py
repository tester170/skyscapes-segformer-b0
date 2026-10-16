"""Shared helpers of the bundle: paths inside the bundle and the config with run-time data paths."""
import os
import sys
from pathlib import Path

# Notebook kernels export MPLBACKEND=module://matplotlib_inline...; that module is not in this environment.
os.environ['MPLBACKEND'] = 'Agg'

BUNDLE = Path(__file__).resolve().parents[1]
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

CONFIG = BUNDLE / 'configs' / 'segformer_b0_skyscapes.py'
MERGES = BUNDLE / 'splits' / 'merge_ladder.json'
CKPT_BEST = BUNDLE / 'checkpoints' / 'segformer_b0_skyscapes_ep220.pth'
CKPT_ENCODER = BUNDLE / 'checkpoints' / 'mit_b0_loveda_encoder.pth'


def load_cfg(cache_root=None, merges=MERGES):
    """Bundle config with the decoded-frame cache (built by prepare_data.py) plugged in."""
    from mmengine.config import Config
    cfg = Config.fromfile(str(CONFIG))
    for dl in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        ds = cfg[dl]['dataset']
        if cache_root:
            ds['cache_root'] = str(cache_root)
            ds['folds_file'] = str(Path(cache_root) / 'folds.json')
    for ev in ('val_evaluator', 'test_evaluator'):
        cfg[ev]['merges_file'] = str(merges)
    return cfg


def cap_vram(margin_gb=1.5):
    """Hard per-process VRAM cap: an overflow becomes an OOM error instead of a silent slowdown."""
    import torch
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(free - margin_gb * 2 ** 30, 2 ** 30) / total)
    print(f'GPU {torch.cuda.get_device_name()}: free {free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GB', flush=True)


def cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
