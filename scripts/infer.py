"""Full-size semantic masks for SkyScapes-like aerial rasters with a trained MMSeg checkpoint.

Input: a file or a folder. JPG/JPEG/PNG/BMP/WEBP are decoded exactly as in training (cv2 via mmcv);
GeoTIFF/TIFF/JP2/IMG/VRT (any GDAL raster) are read with rasterio window by window, so large scenes are never
loaded whole. 8-bit RGB is expected (--bands picks the RGB bands, default 1 2 3); other bit depths are stretched to
8 bit (2-98 % per band) with a warning, because the model was trained on 8-bit camera images.

Output per input <stem> (GeoTIFF with the input CRS / transform and an embedded colour table for georeferenced
inputs, PNG otherwise; --format overrides):
  <stem>_classes   uint8 class codes 1..20 (same coding as SS_Dense labels/grayscale), 0 = no data
  <stem>_color     colour mask in the SkyScapes palette (as labels/rgb)
  <stem>_conf      optional (--conf): maximum class probability x 255
  <stem>_L<n>_classes / _L<n>_color   optional (--schemes 10 9 6 5): merged class schemes decided by the summed
                   probabilities of each group (splits/merge_ladder.json); legend in schemes_legend.json
  <stem>_preview.jpg   image and colour mask side by side (reduced)
With --labels <folder with <stem>.png|.tif in codes 1..20>, all metrics and confusion matrices are computed
(<out>/eval.json, same format as scripts/evaluate.py; --plot draws the figures).
Inference = validation protocol: normalisation, sliding window 1024 / stride 768 at full resolution, fp32.
Usage: python scripts/infer.py --input <file|folder> --out <dir> [--checkpoint ckpt.pth] [--schemes 9 6 5] [--conf]"""
import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import common
import cv2
import mmcv
import numpy as np
import torch

CV_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
GDAL_EXT = {'.tif', '.tiff', '.jp2', '.img', '.vrt'}

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True, help='image file or folder')
ap.add_argument('--out', required=True)
ap.add_argument('--checkpoint', default=str(common.CKPT_BEST))
ap.add_argument('--schemes', type=int, nargs='*', default=[], help='merged schemes by number of classes, e.g. 10 9 6 5')
ap.add_argument('--conf', action='store_true', help='also save the confidence map')
ap.add_argument('--bands', type=int, nargs=3, default=[1, 2, 3], help='RGB band numbers of GDAL rasters (1-based)')
ap.add_argument('--format', choices=['auto', 'tif', 'png'], default='auto',
                help='auto: GeoTIFF for georeferenced inputs, PNG otherwise')
ap.add_argument('--labels', help='folder with reference labels <stem>.png/.tif (codes 1..20) -> metrics')
ap.add_argument('--plot', action='store_true', help='with --labels: draw confusion-matrix figures')
ap.add_argument('--max-mpix', type=float, default=40.0, help='larger rasters are processed in blocks')
ap.add_argument('--block', type=int, default=4096)
ap.add_argument('--margin', type=int, default=512)
a = ap.parse_args()

from mmengine.runner import load_checkpoint  # noqa: E402
from mmseg.registry import MODELS  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402
register_all_modules(init_default_scope=True)
import skyscapes_mmseg  # noqa: E402,F401
from skyscapes_mmseg.data import PALETTE  # noqa: E402
from skyscapes_mmseg.seg_metrics import load_chains, per_class_and_overall  # noqa: E402


# ---------------------------------------------------------------- readers
class CvImage:
    """Whole image decoded as in training (mmcv, cv2 backend, BGR)."""
    georef = False

    def __init__(self, path):
        self.img = mmcv.imfrombytes(Path(path).read_bytes(), flag='color', backend='cv2')
        self.shape = self.img.shape[:2]

    def read(self, y0, y1, x0, x1):
        return self.img[y0:y1, x0:x1]

    def thumbnail(self, width):
        k = width / self.shape[1]
        return cv2.resize(self.img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


class GdalImage:
    """GDAL raster read window by window (rasterio); returns BGR uint8 like the training decoder."""

    def __init__(self, path, bands):
        import rasterio
        self.ds = rasterio.open(path)
        self.bands = list(bands) if self.ds.count >= 3 else [1, 1, 1]
        if self.ds.count < 3:
            warnings.warn(f'{path}: {self.ds.count} band(s); band 1 is repeated as R, G, B')
        self.shape = (self.ds.height, self.ds.width)
        self.georef = self.ds.crs is not None or not self.ds.transform.is_identity
        self.scale = None
        if self.ds.dtypes[self.bands[0] - 1] != 'uint8':
            step = max(1, int(np.sqrt(self.ds.width * self.ds.height / 4e6)))
            s = self.ds.read(self.bands, out_shape=(3, self.ds.height // step, self.ds.width // step)).astype(np.float64)
            lo, hi = np.percentile(s.reshape(3, -1), [2, 98], axis=1)
            self.scale = (lo[:, None, None], np.maximum(hi - lo, 1e-6)[:, None, None])
            warnings.warn(f'{path}: {self.ds.dtypes[0]} data stretched to 8 bit (2-98 %); the model was trained on 8-bit RGB')

    def _to_bgr(self, rgb):
        if self.scale is not None:
            rgb = np.clip((rgb.astype(np.float64) - self.scale[0]) / self.scale[1] * 255, 0, 255)
        return np.ascontiguousarray(rgb.astype(np.uint8).transpose(1, 2, 0)[..., ::-1])

    def read(self, y0, y1, x0, x1):
        from rasterio.windows import Window
        return self._to_bgr(self.ds.read(self.bands, window=Window(x0, y0, x1 - x0, y1 - y0)))

    def thumbnail(self, width):
        h = max(1, round(self.shape[0] * width / self.shape[1]))
        return self._to_bgr(self.ds.read(self.bands, out_shape=(3, h, width)))


def open_image(path):
    return GdalImage(path, a.bands) if Path(path).suffix.lower() in GDAL_EXT else CvImage(path)


# ---------------------------------------------------------------- writers
def write_mask(stem_path, arr, src, palette=None, nodata=0):
    """Single-band uint8 mask (codes) or, with arr.ndim == 3, an RGB image; GeoTIFF keeps the input georeference."""
    fmt = a.format if a.format != 'auto' else ('tif' if isinstance(src, GdalImage) and src.georef else 'png')
    if fmt == 'png':
        img = arr if arr.ndim == 2 else arr[..., ::-1]
        assert cv2.imwrite(str(stem_path) + '.png', img), stem_path
        return
    import rasterio
    prof = dict(driver='GTiff', height=arr.shape[0], width=arr.shape[1], count=1 if arr.ndim == 2 else 3,
                dtype='uint8', compress='deflate', tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    if isinstance(src, GdalImage):
        prof.update(crs=src.ds.crs, transform=src.ds.transform)
    if arr.ndim == 2:
        prof['nodata'] = nodata
    with rasterio.open(str(stem_path) + '.tif', 'w', **prof) as dst:
        if arr.ndim == 2:
            dst.write(arr, 1)
            if palette is not None:
                dst.write_colormap(1, {i: tuple(int(v) for v in palette[i]) + (255,) for i in range(256)})
        else:
            dst.write(arr.transpose(2, 0, 1))


def read_label(path):
    if path.suffix.lower() in GDAL_EXT:
        import rasterio
        with rasterio.open(path) as ds:
            return ds.read(1).astype(np.int64)
    return mmcv.imfrombytes(path.read_bytes(), flag='unchanged', backend='pillow').squeeze().astype(np.int64)


# ---------------------------------------------------------------- model
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
common.cap_vram()
cfg = common.load_cfg()
model = MODELS.build(cfg.model)
load_checkpoint(model, a.checkpoint, map_location='cpu', logger='silent')
model = model.cuda().eval()
classes = json.loads((common.BUNDLE / 'splits' / 'classes.json').read_text(encoding='utf-8'))
pal = np.zeros((256, 3), np.uint8)
pal[1:21] = PALETTE                                            # codes 1..20 -> RGB
chains, _, names0 = load_chains(common.MERGES)
chain_name = next(iter(chains))
ladder = {n: (G, names) for n, G, names in chains[chain_name]}
schemes = {}
for n in a.schemes:
    assert n in ladder, f'no {n}-class scheme; available: {sorted(ladder)}'
    G, names = ladder[n]
    share = np.asarray(classes['train_share'])
    gpal = np.zeros((256, 3), np.uint8)
    for j in range(G.shape[1]):                                # group colour = colour of its most frequent class
        members = np.flatnonzero(G[:, j])
        gpal[j + 1] = PALETTE[members[np.argmax(share[members])]]
    schemes[n] = dict(G=torch.as_tensor(G, dtype=torch.float32, device='cuda'), names=names, pal=gpal)
legend = {'L20': {str(i + 1): nm for i, nm in enumerate(classes['names_ru'])}}
legend.update({f'L{n}': {str(j + 1): nm for j, nm in enumerate(s['names'])} for n, s in schemes.items()})
(out / 'schemes_legend.json').write_text(json.dumps(legend, ensure_ascii=False, indent=1), encoding='utf-8')


@torch.no_grad()
def logits_of(img_bgr):
    x = torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1)
    inp = model.data_preprocessor(dict(inputs=[x]), False)['inputs']
    h, w = inp.shape[2:]
    meta = [dict(ori_shape=(h, w), img_shape=(h, w), pad_shape=(h, w), padding_size=[0, 0, 0, 0])]
    return model.slide_inference(inp, meta)[0]                # C x h x w, fp32


def predict(src):
    H, W = src.shape
    pred = np.zeros((H, W), np.uint8)
    conf = np.zeros((H, W), np.uint8) if a.conf else None
    spred = {n: np.zeros((H, W), np.uint8) for n in schemes}
    if H * W <= a.max_mpix * 1e6:
        blocks = [(0, H, 0, W)]
    else:
        blocks = [(y, min(y + a.block, H), x, min(x + a.block, W)) for y in range(0, H, a.block) for x in range(0, W, a.block)]
    for y0, y1, x0, x1 in blocks:
        ys, ye = max(0, y0 - a.margin), min(H, y1 + a.margin)
        xs, xe = max(0, x0 - a.margin), min(W, x1 + a.margin)
        lg = logits_of(src.read(ys, ye, xs, xe))[:, y0 - ys:y1 - ys, x0 - xs:x1 - xs]
        pred[y0:y1, x0:x1] = (lg.argmax(0) + 1).byte().cpu().numpy()
        if conf is not None or schemes:
            p = torch.softmax(lg, 0)
            if conf is not None:
                conf[y0:y1, x0:x1] = (p.max(0).values * 255).round().byte().cpu().numpy()
            for n, s in schemes.items():
                q = torch.einsum('cg,chw->ghw', s['G'], p)
                spred[n][y0:y1, x0:x1] = (q.argmax(0) + 1).byte().cpu().numpy()
        del lg
    return pred, conf, spred


src_path = Path(a.input)
exts = CV_EXT | GDAL_EXT
files = sorted(p for p in src_path.iterdir() if p.suffix.lower() in exts) if src_path.is_dir() else [src_path]
assert files, f'no images in {src_path}'
ev = dict(checkpoint=a.checkpoint, frames={}, levels_cm={})
for f in files:
    t0 = time.perf_counter()
    src = open_image(f)
    pred, conf, spred = predict(src)
    write_mask(out / f'{f.stem}_classes', pred, src, pal)
    write_mask(out / f'{f.stem}_color', pal[pred], src)
    if conf is not None:
        write_mask(out / f'{f.stem}_conf', conf, src, nodata=None)
    for n, sp in spred.items():
        write_mask(out / f'{f.stem}_L{n}_classes', sp, src, schemes[n]['pal'])
        write_mask(out / f'{f.stem}_L{n}_color', schemes[n]['pal'][sp], src)
    small = src.thumbnail(1400)
    cmask = cv2.resize(pal[pred][..., ::-1], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / f'{f.stem}_preview.jpg'), np.concatenate([small, cmask], 1), [cv2.IMWRITE_JPEG_QUALITY, 90])
    msg = f'{f.name}: {src.shape[1]}x{src.shape[0]}{" (georeferenced)" if src.georef else ""} -> masks in {time.perf_counter() - t0:.1f} s'
    if a.labels:
        cands = [Path(a.labels) / f'{f.stem}{e}' for e in ('.png', '.tif', '.tiff')]
        lp = next((c for c in cands if c.exists()), None)
        if lp is not None:
            gt = read_label(lp)
            m = (gt >= 1) & (gt <= 20)
            cm = np.bincount((gt[m] - 1) * 20 + (pred[m].astype(np.int64) - 1), minlength=400).reshape(20, 20)
            ev['frames'][f.stem] = cm.tolist()
            for n, sp in spred.items():
                G = ladder[n][0]
                g = G.argmax(1)[gt[m] - 1]
                cmn = np.bincount(g * n + (sp[m].astype(np.int64) - 1), minlength=n * n).reshape(n, n)
                key = f'{chain_name}{n}'
                ev['levels_cm'][key] = (np.asarray(ev['levels_cm'].get(key, np.zeros((n, n), np.int64))) + cmn).tolist()
            msg += f' | mIoU(20) {100 * per_class_and_overall(cm)[1]["mIoU"]:.2f}'
        else:
            msg += ' | no label found'
    print(msg, flush=True)
if ev['frames']:
    cm = np.sum([np.asarray(c) for c in ev['frames'].values()], 0)
    ov = per_class_and_overall(cm)[1]
    ev['summary'] = dict(mIoU=100 * ov['mIoU'], kappa=ov['Kappa'], OA=100 * ov['OA'], mDice=100 * ov['mDice'])
    for key, c in ev['levels_cm'].items():
        ev['summary'][f'mIoU_{key}'] = 100 * per_class_and_overall(np.asarray(c))[1]['mIoU']
    (out / 'eval.json').write_text(json.dumps(ev))
    print('all labelled images:', {k: round(v, 4) for k, v in ev['summary'].items()})
    if a.plot:
        subprocess.run([sys.executable, str(common.BUNDLE / 'scripts' / 'plot_confusion.py'), str(out / 'eval.json'),
                        str(out / 'figures'), '--merges', str(common.MERGES), '--chain-title', 'none',
                        '--title-prefix', 'SegFormer-B0'], check=True)
