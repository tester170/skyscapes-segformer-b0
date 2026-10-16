"""Download the large files of the project from its GitHub release and verify their SHA-256 (standard library only).

  model    trained SegFormer-B0 checkpoint (epoch 220)      -> checkpoints/segformer_b0_skyscapes_ep220.pth
  encoder  LoveDA-pretrained MiT-B0 encoder, MMSeg format   -> checkpoints/mit_b0_loveda_encoder.pth
  mmcv     MMCV 2.1.0 wheel (torch 2.1, CUDA 12.1, cp311)   -> official OpenMMLab build; OpenMMLab server as fallback
  dataset  SS_Dense of DLR SkyScapes (train/val/test)       -> <out>/SS_Dense; official DLR server as fallback
Repository, tag, file names and checksums are in env/assets.json. Files already present with the right checksum are
not downloaded again.
Usage: python scripts/fetch.py model|encoder|mmcv [--out FILE] | python scripts/fetch.py dataset --out FOLDER"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
ASSETS = json.loads((BUNDLE / 'env' / 'assets.json').read_text(encoding='utf-8'))
DEFAULT_OUT = {'model': BUNDLE / 'checkpoints' / 'segformer_b0_skyscapes_ep220.pth',
               'encoder': BUNDLE / 'checkpoints' / 'mit_b0_loveda_encoder.pth',
               'mmcv': BUNDLE / 'wheels' / 'mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dst, expected=None, tries=3):
    """Download with progress and retries; expected=None means the file is checked afterwards in another way."""
    dst = Path(dst)
    if dst.exists() and (expected is None or sha256(dst) == expected):
        print(f'{dst.name}: already present', flush=True)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + '.part')
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'skyscapes-segformer'})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, 'wb') as f:
                total, done, t0, shown = int(r.headers.get('Content-Length') or 0), 0, time.time(), -10
                for chunk in iter(lambda: r.read(1 << 22), b''):
                    f.write(chunk)
                    done += len(chunk)
                    pct = 100 * done // total if total else 0
                    if pct >= shown + 10 or done == total:
                        shown = pct
                        print(f'  {dst.name}: {done / 2 ** 20:.0f}/{total / 2 ** 20:.0f} MB '
                              f'({done / 2 ** 20 / max(time.time() - t0, 1e-3):.1f} MB/s)', flush=True)
            got = sha256(part)
            if expected and got != expected:
                raise IOError(f'checksum mismatch: {got} (expected {expected})')
            os.replace(part, dst)
            return dst
        except Exception as e:  # noqa: BLE001
            print(f'  attempt {attempt}/{tries} failed: {e}', flush=True)
            part.unlink(missing_ok=True)
            if attempt == tries or 400 <= getattr(e, 'code', 0) < 500:   # missing file: no point in retrying
                raise
            time.sleep(5 * attempt)


def from_release(key, dst):
    a = ASSETS['release'][key]
    url = f'https://github.com/{ASSETS["repo"]}/releases/download/{ASSETS["tag"]}/{a["name"]}'
    print(f'{a["name"]} ({a["size"] / 2 ** 20:.0f} MB) from {ASSETS["repo"]} release {ASSETS["tag"]}', flush=True)
    return download(url, dst, a['sha256'])


def extract_ss_dense(zf, root):
    for n in zf.namelist():
        if n.startswith('SS_Dense/') and not n.endswith('/'):
            p = root / n
            p.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(n) as src, open(p, 'wb') as dst:
                shutil.copyfileobj(src, dst)


def dataset(root):
    root = Path(root)
    if all((root / 'SS_Dense' / s / 'images').is_dir() for s in ('train', 'val')):
        print(f'dataset already present: {root / "SS_Dense"}', flush=True)
        return
    try:
        z = from_release('dataset', root / ASSETS['release']['dataset']['name'])
        with zipfile.ZipFile(z) as zf:
            extract_ss_dense(zf, root)
    except Exception as e:  # noqa: BLE001
        d = ASSETS['dataset_official']
        print(f'release unavailable ({e}); official DLR archive {d["url"]}', flush=True)
        z = download(d['url'], root / d['name'], d['sha256'])
        with zipfile.ZipFile(z) as outer, zipfile.ZipFile(io.BytesIO(outer.read(d['inner']))) as zf:
            extract_ss_dense(zf, root)
    z.unlink()
    n_img = {s: len(list((root / 'SS_Dense' / s / 'images').glob('*'))) for s in ('train', 'val', 'test')}
    print(f'dataset ready: {root / "SS_Dense"} (images {n_img})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('model', 'encoder', 'mmcv', 'dataset'))
    ap.add_argument('--out', help='target file (model, encoder, mmcv) or folder that receives SS_Dense (dataset)')
    a = ap.parse_args()
    if a.what == 'dataset':
        dataset(a.out or BUNDLE.parent / 'data')
        return
    out = Path(a.out) if a.out else DEFAULT_OUT[a.what]
    if a.what != 'mmcv':
        from_release(a.what, out)
        return
    try:
        from_release('mmcv', out)
    except Exception as e:  # noqa: BLE001
        print(f'release unavailable ({e}); official OpenMMLab wheel {ASSETS["mmcv_official"]}', flush=True)
        download(ASSETS['mmcv_official'], out)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print('FETCH FAILED:', e, file=sys.stderr)
        sys.exit(1)
