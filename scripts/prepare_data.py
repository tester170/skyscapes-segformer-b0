"""Decode the SkyScapes-Dense train/val frames once into a .npy cache (fast local disk, e.g. /content/cache) and
write folds.json with the official split (train -> fit, val -> evaluation).

Cache = exactly what MMSeg LoadImageFromFile (cv2, BGR) + LoadAnnotations(reduce_zero_label=True) would produce:
  frames/<stem>.img.npy  uint8 HxWx3 BGR;  frames/<stem>.lbl.npy  uint8 HxW train IDs 0..19 (255 = no label);
  index/<stem>.cls.npz   class anchor points (rare-class patches);  manifest.json.
Idempotent: frames already in the manifest are skipped.
Usage: python scripts/prepare_data.py --data <.../SS_Dense> --cache /content/cache"""
import argparse
import json
import os
import time
from pathlib import Path

import common  # noqa: F401  (bundle on sys.path)
import mmcv
import numpy as np

K, STRIDE = 20, 4

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, help='SS_Dense folder with train/ and val/ (images/, labels/grayscale/)')
ap.add_argument('--cache', required=True)
a = ap.parse_args()
src, dst = Path(a.data), Path(a.cache)
(dst / 'frames').mkdir(parents=True, exist_ok=True)
(dst / 'index').mkdir(parents=True, exist_ok=True)
man_path = dst / 'manifest.json'
man = json.loads(man_path.read_text()) if man_path.exists() else dict(version='skyscapes_dense_cache_v1', frames={})


def save(path, arr):
    tmp = path.with_suffix('.tmp.npy')
    np.save(tmp, arr)
    os.replace(tmp, path)


splits = {}
for split in ('train', 'val'):
    imgs = sorted((src / split / 'images').glob('*.jpg'))
    assert imgs, f'no images in {src / split / "images"}'
    splits[split] = [p.stem for p in imgs]
    for p in imgs:
        stem = p.stem
        lbl_p = src / split / 'labels' / 'grayscale' / f'{stem}.png'
        files = [dst / 'frames' / f'{stem}.img.npy', dst / 'frames' / f'{stem}.lbl.npy', dst / 'index' / f'{stem}.cls.npz']
        if stem in man['frames'] and all(f.exists() for f in files):
            continue
        t0 = time.perf_counter()
        img = mmcv.imfrombytes(p.read_bytes(), flag='color', backend='cv2')
        lbl = mmcv.imfrombytes(lbl_p.read_bytes(), flag='unchanged', backend='pillow').squeeze().astype(np.uint8)
        assert img.shape[:2] == lbl.shape, (stem, img.shape, lbl.shape)
        lbl[lbl == 0] = 255
        lbl = lbl - 1
        lbl[lbl == 254] = 255
        counts = np.bincount(lbl.ravel(), minlength=256)
        g = lbl[STRIDE // 2::STRIDE, STRIDE // 2::STRIDE].ravel()
        order = np.argsort(g, kind='stable')
        cls = g[order]
        keep = cls < K
        order, cls = order[keep], cls[keep]
        gw = lbl[STRIDE // 2::STRIDE, STRIDE // 2::STRIDE].shape[1]
        ys, xs = np.divmod(order, gw)
        yx = np.stack([ys * STRIDE + STRIDE // 2, xs * STRIDE + STRIDE // 2], 1).astype(np.uint16)
        offsets = np.searchsorted(cls, np.arange(K + 1)).astype(np.int64)
        save(files[0], np.ascontiguousarray(img))
        save(files[1], lbl)
        tmp = files[2].with_suffix('.tmp.npz')
        np.savez(tmp, yx=yx, offsets=offsets)
        os.replace(tmp, files[2])
        man['frames'][stem] = dict(split=split, shape=list(lbl.shape), class_pixels=counts[:K].tolist())
        tmp = man_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(man, indent=1))
        os.replace(tmp, man_path)
        print(f'{split}/{stem}: cached in {time.perf_counter() - t0:.1f} s', flush=True)
folds = dict(description='fold 0 = official SkyScapes-Dense split (train -> fit, val -> evaluation)',
             folds=[dict(fold=0, train=splits['train'], val=splits['val'])])
(dst / 'folds.json').write_text(json.dumps(folds, indent=1))
print(f'cache ready: {dst} ({len(splits["train"])} train + {len(splits["val"])} val frames)')
