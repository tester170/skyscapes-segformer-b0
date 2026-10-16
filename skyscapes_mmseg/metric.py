"""Confusion-matrix metric for MMSegmentation: one torch.bincount per frame on the prediction device; the per-frame
20x20 matrices are kept (self.last_frames) and every metric of every merge level is derived from them in closed form
(seg_metrics.evaluate_all). The Runner receives a compact set (L0, aliases L1..L5, chain curves, K60); the complete
flat set (mIoU and kappa at every step of every chain) is kept in self.last_flat for the progress display and the
per-validation JSON written by PlateauControlHook."""
from pathlib import Path

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmseg.registry import METRICS

from .seg_metrics import evaluate_all, level_key, load_chains, per_class_and_overall


def metrics_from_confusion(cm):
    """Back-compatible subset used by tests and tools: IoU, Acc, Fscore per class and aAcc."""
    pc, ov = per_class_and_overall(cm)
    return dict(IoU=pc['IoU'], Acc=pc['Recall'], Fscore=pc['Dice'], aAcc=ov['OA'])


COMPACT = ('aAcc', 'mIoU', 'mDice', 'mPrecision', 'mRecall', 'kappa', 'MCC', 'FWIoU')


@METRICS.register_module()
class ConfusionIoUMetric(BaseMetric):
    default_prefix = None

    def __init__(self, ignore_index=255, merges_file=None, target=60.0, merge_scores='softmax', chunk=4_000_000,
                 collect_device='cpu', prefix=None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index, self.target = ignore_index, target
        self.chains, self.levels = ({}, {}) if merges_file is None else load_chains(merges_file)[:2]
        # Merged levels are scored by argmax of the summed class probabilities of each group (the decision a model
        # makes for the merged label set), not by mapping the 20-class argmax. 'softmax': seg_logits are logits
        # (EncoderDecoder heads); 'identity': already probability-like scores (Mask2Former); None: hard mapping.
        self.merge_scores, self.chunk = merge_scores, chunk
        uniq = {}
        for steps in self.chains.values():
            for n, G, _ in steps[1:]:
                uniq.setdefault(level_key(G), G)
        self.level_G = uniq
        self.last_frames = self.last_flat = self.last_level_cms = None
        self._dev_G = None

    def _level_confusions(self, logits, gt):
        """({level_key: n x n confusion} by probability-sum merging, (sum of pixel cross-entropy, pixel count)).
        GPU matmul per level over pixel chunks; the validation cross-entropy (same loss as training, from the same
        softmax) is accumulated in the same pass, so it costs no extra pass over the frame."""
        dev = logits.device
        if self._dev_G is None or self._dev_G[0] != dev:
            mats = [(k, torch.as_tensor(G, dtype=torch.float32, device=dev),
                     torch.as_tensor(G.argmax(1), dtype=torch.long, device=dev)) for k, G in self.level_G.items()]
            self._dev_G = (dev, mats)
        mats = self._dev_G[1]
        C = logits.shape[0]
        flat_l, flat_g = logits.reshape(C, -1), gt.reshape(-1)
        acc = {k: torch.zeros(G.shape[1] ** 2, dtype=torch.long, device=dev) for k, G, _ in mats}
        ce_sum, ce_n = torch.zeros((), dtype=torch.float64, device=dev), 0
        for s in range(0, flat_g.numel(), self.chunk):
            g = flat_g[s:s + self.chunk].long()
            valid = g != self.ignore_index
            if not bool(valid.any()):
                continue
            p = flat_l[:, s:s + self.chunk][:, valid].float()
            g = g[valid]
            if self.merge_scores == 'softmax':
                logp = torch.log_softmax(p, 0)
                ce_sum += -logp.gather(0, g[None]).sum(dtype=torch.float64)
                ce_n += int(g.numel())
                p = logp.exp_()
            for k, G, cls2grp in mats:
                n = G.shape[1]
                pred = (G.t() @ p).argmax(0)
                acc[k] += torch.bincount(cls2grp[g] * n + pred, minlength=n * n)
        return ({k: v.reshape(int(v.numel() ** 0.5), -1).cpu().numpy() for k, v in acc.items()},
                (float(ce_sum), ce_n))

    def process(self, data_batch, data_samples):
        k = len(self.dataset_meta['classes'])
        for ds in data_samples:
            pred = ds['pred_sem_seg']['data'].squeeze()
            gt = ds['gt_sem_seg']['data'].squeeze().to(pred.device)
            m = gt != self.ignore_index
            cm = torch.bincount(gt[m].long() * k + pred[m].long(), minlength=k * k).reshape(k, k)
            lv, ce = None, (0.0, 0)
            if self.merge_scores and self.level_G and 'seg_logits' in ds:
                lv, ce = self._level_confusions(ds['seg_logits']['data'], gt)
            self.results.append((Path(ds['img_path']).name.split('.')[0], cm.cpu().numpy(), lv, ce))

    def compute_metrics(self, results):
        classes = self.dataset_meta['classes']
        self.last_frames = {r[0]: r[1] for r in results}
        cm = np.sum([r[1] for r in results], 0)
        level_cms = None
        if results and all(r[2] is not None for r in results):
            level_cms = {key: np.sum([r[2][key] for r in results], 0) for key in results[0][2]}
        ce_n = sum(r[3][1] for r in results)
        self.last_level_cms = level_cms
        flat, full = evaluate_all(cm, self.chains, self.levels, self.target, level_cms)
        if ce_n:
            flat['val_loss'] = sum(r[3][0] for r in results) / ce_n  # pixel-mean cross-entropy, as the training loss
        self.last_flat = flat
        pc = full['L0']['per_class']
        rows = '\n'.join(f'{n:>24s} IoU {100 * pc["IoU"][i]:6.2f}  Dice {100 * pc["Dice"][i]:6.2f}  '
                         f'P {100 * pc["Precision"][i]:6.2f}  R {100 * pc["Recall"][i]:6.2f}  '
                         f'kappa {pc["Kappa"][i]:5.3f}' for i, n in enumerate(classes))
        logger = MMLogger.get_current_instance()
        print_log('per-class results (20 classes):\n' + rows, logger)
        for c in self.chains:
            print_log(f'chain {c} mIoU by number of classes: ' + ' '.join(
                f'{n}:{flat[f"mIoU_{c}{n}"]:.1f}' for n, _, _ in self.chains[c]), logger)
        keys = list(COMPACT) + (['val_loss'] if 'val_loss' in flat else []) + [k for k in flat if k[-3:-1] == '_L' or k.endswith('_curve') or k.startswith('K')]
        keys = [k for k in keys if not k.startswith(('mDice_L', 'mPrecision_L', 'mRecall_L', 'FWIoU_L'))]
        return {k: flat[k] for k in keys}
