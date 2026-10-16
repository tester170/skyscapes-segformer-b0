"""Verifies the environment: exact package versions, GPU, compiled MMCV CUDA ops, Numba, bundle modules.
Exit code 0 only if everything matches. Usage: python env/check_env.py [--quiet]"""
import os
import sys
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'  # notebook kernels export an inline backend that this environment does not have

QUIET = '--quiet' in sys.argv
EXPECTED = {'torch': '2.1.0+cu121', 'torchvision': '0.16.0+cu121', 'mmcv': '2.1.0', 'mmengine': '0.10.7',
            'mmseg': '1.2.2', 'numpy': '1.26.4', 'numba': '0.60.0', 'cv2': '4.11.0', 'rasterio': '1.4.3'}


def say(*a):
    if not QUIET:
        print(*a, flush=True)


try:
    import importlib
    bad = []
    for mod, ver in EXPECTED.items():
        v = importlib.import_module(mod).__version__
        say(f'{mod:12s} {v}')
        if v != ver:
            bad.append(f'{mod} {v} != {ver}')
    assert sys.version_info[:2] == (3, 11), f'Python {sys.version} (expected 3.11)'
    assert not bad, '; '.join(bad)
    import torch
    assert torch.cuda.is_available(), 'CUDA GPU not available (Colab: Runtime -> Change runtime type -> GPU)'
    say(f'GPU          {torch.cuda.get_device_name()}  (CUDA {torch.version.cuda})')
    from mmcv.ops import nms, point_sample
    b = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=torch.float32, device='cuda')
    nms(b, torch.tensor([0.9, 0.8], device='cuda'), 0.5)
    assert torch.isfinite(point_sample(torch.rand(1, 2, 4, 4, device='cuda'), torch.rand(1, 3, 2, device='cuda'))).all()
    say('mmcv CUDA ops ok')
    from numba import njit
    assert njit(lambda x: x + 1)(1) == 2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import skyscapes_mmseg  # noqa: F401
    say('bundle modules ok')
    say('ENVIRONMENT OK')
except Exception as e:  # noqa: BLE001
    if not QUIET:
        print('ENVIRONMENT CHECK FAILED:', e)
    sys.exit(1)
