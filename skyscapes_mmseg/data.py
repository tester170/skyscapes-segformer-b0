"""SkyScapes-Dense datasets and transforms for MMSegmentation 1.2.2 (registered in mmseg registries).

Frames come from the decoded cache built by tools/e0_build_cache.py (BGR uint8 image, train-ID label 0..19/255),
opened lazily with np.load(mmap_mode='r') so dataloader workers share page-cache pages.

SkyScapesFrames      full frames of one fold part (validation / CV evaluation).
SkyScapesCrops       training crops sampled directly from the frames: anchor pixel -> scale -> window -> resize,
                     so a full 21 Mpx frame is never decoded or resized per sample.
RandomDihedral       random rot90 + flip (the 8 symmetries of the square; nadir views have no preferred orientation).
"""
import json
from pathlib import Path

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS

CLASSES = ('low vegetation', 'paved road', 'non paved road', 'paved parking place', 'non paved parking place',
           'bikeways', 'sidewalks', 'entrance exit', 'danger area', 'lane-markings', 'building', 'car', 'trailer',
           'van', 'truck', 'long truck', 'bus', 'clutter', 'impervious surface', 'tree')
PALETTE = [[244, 35, 232], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [152, 251, 152],
           [70, 130, 180], [107, 142, 35], [220, 220, 0], [255, 0, 0], [220, 20, 60], [125, 0, 142], [170, 200, 40],
           [200, 60, 100], [150, 18, 80], [81, 180, 81], [190, 241, 21], [11, 119, 32], [120, 240, 120], [70, 70, 70]]
K = len(CLASSES)


def _fold_frames(folds_file, fold, split):
    folds = json.loads(Path(folds_file).read_text())['folds']
    return list(folds[fold][split])


class _CacheMixin:
    """Lazy per-process memmaps of cached frames."""

    def _open(self, stem):
        if self._mm is None:
            self._mm = {}
        if stem not in self._mm:
            root = Path(self.cache_root) / 'frames'
            self._mm[stem] = (np.load(root / f'{stem}.img.npy', mmap_mode='r'),
                              np.load(root / f'{stem}.lbl.npy', mmap_mode='r'))
        return self._mm[stem]


@DATASETS.register_module()
class SkyScapesFrames(_CacheMixin, BaseSegDataset):
    METAINFO = dict(classes=CLASSES, palette=PALETTE)

    def __init__(self, cache_root, folds_file, fold, split, **kwargs):
        self.cache_root, self.folds_file, self.fold, self.split = cache_root, folds_file, fold, split
        self._mm = None
        kwargs.setdefault('reduce_zero_label', False)  # cache already holds train IDs
        super().__init__(**kwargs)

    def load_data_list(self):
        root = Path(self.cache_root) / 'frames'
        return [dict(img_path=str(root / f'{s}.img.npy'), seg_map_path=str(root / f'{s}.lbl.npy'), frame=s,
                     label_map=self.label_map, reduce_zero_label=False, seg_fields=[])
                for s in _fold_frames(self.folds_file, self.fold, self.split)]


@TRANSFORMS.register_module()
class LoadCachedFrame(BaseTransform):
    """Full cached frame + label; equals LoadImageFromFile + LoadAnnotations(reduce_zero_label=True) bit-for-bit."""

    def transform(self, results):
        # copy-on-write map: writable for torch.from_numpy without copying the 63 MB frame
        img = np.load(results['img_path'], mmap_mode='c')
        lbl = np.load(results['seg_map_path'], mmap_mode='c')
        results['img'] = np.ascontiguousarray(img)
        results['gt_seg_map'] = np.ascontiguousarray(lbl)
        results['seg_fields'].append('gt_seg_map')
        results['img_shape'] = results['ori_shape'] = img.shape[:2]
        return results


@DATASETS.register_module()
class SkyScapesCrops(_CacheMixin, BaseSegDataset):
    """Infinite-style training dataset; the index is ignored, every item is a fresh random crop.

    Anchor: with probability rcs_prob a rare-class anchor (class c ~ RCS, DAFormer: p_c ∝ exp((1 - f_c) / T),
    f_c = pixel frequency of c in this fold's training frames; then a uniform anchor point of class c), otherwise a
    uniform pixel of a uniform training frame. Scale s ~ U(ratio_range); a source window of round(crop / s) px that
    contains the anchor at a uniform position is cut and resized to crop (bilinear image / nearest label), which is
    mmseg RandomResize + RandomCrop restricted to the window that is actually used.
    """
    METAINFO = dict(classes=CLASSES, palette=PALETTE)

    def __init__(self, cache_root, folds_file, fold, crop_size=(1024, 1024), ratio_range=(0.75, 1.5),
                 rcs_prob=1.0, rcs_temperature=0.01, virtual_length=100000, **kwargs):
        self.cache_root, self.folds_file, self.fold = cache_root, folds_file, fold
        self.crop_size, self.ratio_range = tuple(crop_size), tuple(ratio_range)
        self.rcs_prob, self.rcs_temperature, self.virtual_length = rcs_prob, rcs_temperature, virtual_length
        self._mm = None
        kwargs.setdefault('reduce_zero_label', False)
        super().__init__(**kwargs)

    def load_data_list(self):
        self.frames = _fold_frames(self.folds_file, self.fold, 'train')
        man = json.loads((Path(self.cache_root) / 'manifest.json').read_text())['frames']
        idx = Path(self.cache_root) / 'index'
        pix = np.zeros(K, np.int64)
        per_cls_fid, per_cls_yx = [[] for _ in range(K)], [[] for _ in range(K)]
        for fid, s in enumerate(self.frames):
            pix += np.asarray(man[s]['class_pixels'], np.int64)
            z = np.load(idx / f'{s}.cls.npz')
            yx, off = z['yx'], z['offsets']
            for c in range(K):
                per_cls_yx[c].append(yx[off[c]:off[c + 1]])
                per_cls_fid[c].append(np.full(off[c + 1] - off[c], fid, np.uint8))
        self.anchor_yx = [np.concatenate(a) for a in per_cls_yx]
        self.anchor_fid = [np.concatenate(a) for a in per_cls_fid]
        self.shapes = [tuple(man[s]['shape']) for s in self.frames]
        freq = pix / pix.sum()
        logits = np.where(pix > 0, (1 - freq) / self.rcs_temperature, -np.inf)
        p = np.exp(logits - logits.max())
        self.rcs_p = p / p.sum()
        self.class_freq = freq
        return [dict(frame=s) for s in self.frames]

    def __len__(self):
        return self.virtual_length

    def _rcs_anchor(self):
        c = np.random.choice(K, p=self.rcs_p)
        j = np.random.randint(len(self.anchor_yx[c]))
        ay, ax = (int(v) for v in self.anchor_yx[c][j])
        return int(self.anchor_fid[c][j]), (ay, ay + 1, ax, ax + 1), c

    def _window_sample(self, fid, box, anchor_class):
        """Scale s ~ U(ratio_range); a source window of round(crop / s) px containing `box` (y0, y1, x0, x1) at a
        uniform position is cut and resized to the crop size (image: area when shrinking, bilinear when enlarging;
        label: nearest)."""
        ch, cw = self.crop_size
        by0, by1, bx0, bx1 = box
        s = np.random.uniform(*self.ratio_range)
        H, W = self.shapes[fid]
        wh, ww = min(int(round(ch / s)), H), min(int(round(cw / s)), W)
        y0 = np.random.randint(max(by1 - wh, 0), min(by0, H - wh) + 1)
        x0 = np.random.randint(max(bx1 - ww, 0), min(bx0, W - ww) + 1)
        img_mm, lbl_mm = self._open(self.frames[fid])
        img = img_mm[y0:y0 + wh, x0:x0 + ww]
        lbl = lbl_mm[y0:y0 + wh, x0:x0 + ww]
        if (wh, ww) != (ch, cw):
            # area averaging when the window is shrunk (anti-aliasing for 1-3 px lane-markings), bilinear when enlarged
            flag = cv2.INTER_AREA if wh > ch else cv2.INTER_LINEAR
            img = cv2.resize(img, (cw, ch), interpolation=flag)
            lbl = cv2.resize(lbl, (cw, ch), interpolation=cv2.INTER_NEAREST)
        else:
            img, lbl = np.array(img), np.array(lbl)
        results = dict(img=img, gt_seg_map=lbl, seg_fields=['gt_seg_map'], img_shape=(ch, cw), ori_shape=(ch, cw),
                       img_path=self.frames[fid], seg_map_path=self.frames[fid], reduce_zero_label=False,
                       anchor_class=anchor_class, crop_box=(fid, y0, x0, wh, ww), scale_factor=(cw / ww, ch / wh))
        return self.pipeline(results)

    def prepare_data(self, idx):
        if not self._fully_initialized:
            self.full_init()
        if np.random.rand() < self.rcs_prob:
            fid, box, c = self._rcs_anchor()
        else:
            fid = np.random.randint(len(self.frames))
            ay, ax = np.random.randint(self.shapes[fid][0]), np.random.randint(self.shapes[fid][1])
            box, c = (ay, ay + 1, ax, ax + 1), -1
        return self._window_sample(fid, box, c)


@DATASETS.register_module()
class SkyScapesTiles(SkyScapesCrops):
    """Epoch-based training set with guaranteed coverage.

    Every training frame is split into a grid of cells no larger than crop / max(ratio_range) (624 x 624 px for
    1024 / 1.5 on 5616 x 3744), so the training window (crop / s px for any s in ratio_range) always contains its
    cell. One epoch = every cell of every frame once, in the sampler's shuffled order (each pixel is trained on at
    least once per epoch, ~2x on average), with a fresh random scale and window offset each time, followed by
    `rcs_fraction` x n_cells extra samples anchored on rare classes (RCS, as SkyScapesCrops)."""

    def __init__(self, *args, rcs_fraction=0.25, **kwargs):
        self.rcs_fraction = rcs_fraction
        super().__init__(*args, rcs_prob=0.0, **kwargs)

    def load_data_list(self):
        data = super().load_data_list()
        cmax = int(np.floor(min(self.crop_size) / max(self.ratio_range)))
        cells = []
        for fid, (H, W) in enumerate(self.shapes):
            ys = np.round(np.linspace(0, H, int(np.ceil(H / cmax)) + 1)).astype(int)
            xs = np.round(np.linspace(0, W, int(np.ceil(W / cmax)) + 1)).astype(int)
            cells += [(fid, ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(len(ys) - 1) for j in range(len(xs) - 1)]
        self.cells = np.asarray(cells, np.int64)
        self.n_tiles = len(cells)
        self.n_rcs = int(round(self.rcs_fraction * self.n_tiles))
        return data

    def __len__(self):
        if not self._fully_initialized:
            self.full_init()
        return self.n_tiles + self.n_rcs

    def prepare_data(self, idx):
        if not self._fully_initialized:
            self.full_init()
        if idx < self.n_tiles:
            fid, y0, y1, x0, x1 = (int(v) for v in self.cells[idx])
            return self._window_sample(fid, (y0, y1, x0, x1), -2)
        fid, box, c = self._rcs_anchor()
        return self._window_sample(fid, box, c)


@TRANSFORMS.register_module()
class RandomDihedral(BaseTransform):
    """Uniform element of the dihedral group D4 (rot90^k, optional flip) on image and seg fields.

    Returns strided views without copying; place it right before PackSegInputs, whose CHW transpose / int64 cast
    makes the single copy. Pixel-wise transforms (photometric) commute with it and go before it."""

    def transform(self, results):
        k, f = np.random.randint(4), np.random.rand() < 0.5
        for key in ['img'] + results.get('seg_fields', []):
            a = np.rot90(results[key], k)
            results[key] = a[:, ::-1] if f else a
        results['img_shape'] = results['img'].shape[:2]
        return results


from mmengine.structures import PixelData  # noqa: E402
from mmseg.datasets.transforms import PackSegInputs  # noqa: E402
import torch  # noqa: E402


@TRANSFORMS.register_module()
class PackSegInputsU8(PackSegInputs):
    """PackSegInputs that keeps the label map as uint8 instead of int64: 8x less pageable host->device traffic per
    batch (67 MB -> 8 MB for 8 x 1024^2). SegDataPreProcessorU8 casts it to int64 on the GPU, so the model sees
    exactly the same tensor."""

    def transform(self, results):
        gt = results.pop('gt_seg_map', None)
        packed = super().transform(results)
        if gt is not None:
            assert gt.ndim == 2 and gt.dtype == np.uint8
            packed['data_samples'].gt_sem_seg = PixelData(data=torch.from_numpy(np.ascontiguousarray(gt)[None]))
        return packed
