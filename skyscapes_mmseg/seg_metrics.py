"""Segmentation metrics from confusion matrices (cm[g, p] = pixels of true class g predicted as p).

All metrics are closed-form array expressions of the confusion matrix, so every merge level costs O(K^2) after the
single GPU bincount of the pixels. Per class (one-vs-rest): IoU, Dice (= F1), precision, recall (= class accuracy),
specificity, Cohen's kappa, MCC, support. Overall: OA, mean of per-class values (nanmean, as mmseg), frequency-weighted
IoU, multi-class Cohen's kappa and multi-class MCC (Gorodkin). Merge chains are built from splits/class_merges.json.
"""
import json
from pathlib import Path

import numpy as np

PER_CLASS = ('IoU', 'Dice', 'Precision', 'Recall', 'Specificity', 'Kappa', 'MCC')


def per_class_and_overall(cm):
    cm = np.asarray(cm, np.float64)
    n = cm.sum()
    tp = np.diag(cm)
    gt, pr = cm.sum(1), cm.sum(0)
    fp, fn = pr - tp, gt - tp
    tn = n - tp - fp - fn
    with np.errstate(divide='ignore', invalid='ignore'):
        pc = dict(
            IoU=tp / (tp + fp + fn),
            Dice=2 * tp / (2 * tp + fp + fn),
            Precision=tp / pr,
            Recall=tp / gt,
            Specificity=tn / (tn + fp),
            # one-vs-rest Cohen's kappa: (po - pe) / (1 - pe) on the 2x2 table of class c
            Kappa=2 * (tp * tn - fp * fn) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)),
            MCC=(tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        )
        pe = (gt * pr).sum() / n ** 2
        po = tp.sum() / n
        cov_tp = tp.sum() * n - (gt * pr).sum()
        overall = dict(
            OA=po,
            Kappa=(po - pe) / (1 - pe),
            MCC=cov_tp / np.sqrt((n ** 2 - (pr ** 2).sum()) * (n ** 2 - (gt ** 2).sum())),
            FWIoU=np.nansum(gt / n * pc['IoU']),
        )
    # A class present in the ground truth but never predicted has precision 0/0: count it as 0 (the model missed the
    # class entirely), so that never predicting rare classes cannot raise mPrecision / mMCC. NaN remains only where
    # the class is absent from both ground truth and prediction.
    missed = (gt > 0) & (pr == 0)
    pc['Precision'] = np.where(missed, 0.0, pc['Precision'])
    pc['MCC'] = np.where(missed, 0.0, pc['MCC'])
    for k in PER_CLASS:
        overall['m' + k] = np.nanmean(pc[k]) if np.isfinite(pc[k]).any() else np.nan
    pc['Support'] = gt
    return pc, overall


def load_chains(path):
    """{chain: [(n_classes, G, names), ...]} from 20 classes down, and {level alias: (chain, n_classes)}."""
    spec = json.loads(Path(path).read_text())
    names0 = spec['classes']
    k = len(names0)
    chains = {}
    for cname, ch in spec['chains'].items():
        group_of = np.arange(k)          # current group id of every original class
        gnames = {i: names0[i] for i in range(k)}
        steps = [(k, np.eye(k, dtype=np.int64), list(names0))]
        for st in ch['steps']:
            ids = {int(group_of[c]) for c in st['merge']}
            assert len(ids) >= 2, f'chain {cname}: step {st} merges nothing new'
            keep = min(ids)
            group_of[np.isin(group_of, list(ids))] = keep
            for g in ids - {keep}:
                gnames.pop(g)
            gnames[keep] = st['name']
            uniq = np.unique(group_of)
            G = (group_of[:, None] == uniq[None, :]).astype(np.int64)
            steps.append((len(uniq), G, [gnames[int(u)] for u in uniq]))
        chains[cname] = steps
    levels = {alias: tuple(v) for alias, v in spec.get('levels', {}).items()}
    return chains, levels, names0


def level_key(G):
    """Identity of a merge level (its group matrix) for de-duplication across chains."""
    return G.tobytes() + bytes(G.shape)


def evaluate_all(cm, chains, levels, target=60.0, level_cms=None):
    """Flat dict of scalar metrics (percent) for logging + nested full results for reports.
    level_cms: {level_key(G): confusion matrix of that level} from probability-sum merging; without it a level is
    scored by hard mapping of the 20-class argmax (G^T C G)."""
    pc0, ov0 = per_class_and_overall(cm)
    flat = {'aAcc': 100 * ov0['OA'], 'mIoU': 100 * ov0['mIoU'], 'mAcc': 100 * ov0['mRecall'],
            'mDice': 100 * ov0['mDice'], 'mPrecision': 100 * ov0['mPrecision'], 'mRecall': 100 * ov0['mRecall'],
            'mKappa_c': 100 * ov0['mKappa'], 'mMCC_c': 100 * ov0['mMCC'], 'FWIoU': 100 * ov0['FWIoU'],
            'kappa': 100 * ov0['Kappa'], 'MCC': 100 * ov0['MCC']}
    full = {'L0': dict(per_class=pc0, overall=ov0)}
    for cname, steps in chains.items():
        curve = []
        for n_cls, G, names in steps:
            if n_cls == steps[0][0]:
                pc, ov = pc0, ov0
            else:
                m = level_cms.get(level_key(G)) if level_cms else None
                pc, ov = per_class_and_overall(G.T @ cm @ G if m is None else m)
            full[f'{cname}{n_cls}'] = dict(per_class=pc, overall=ov, names=names)
            flat[f'mIoU_{cname}{n_cls}'] = 100 * ov['mIoU']
            flat[f'kappa_{cname}{n_cls}'] = 100 * ov['Kappa']
            curve.append((n_cls, 100 * ov['mIoU']))
        flat[f'mIoU_{cname}_curve'] = float(np.mean([v for _, v in curve]))
        ok = [n for n, v in curve if v >= target]
        flat[f'K{int(target)}_{cname}'] = float(max(ok)) if ok else 0.0
    for alias, (cname, n_cls) in levels.items():
        ov = full[f'{cname}{n_cls}']['overall']
        for k in ('mIoU', 'mDice', 'mPrecision', 'mRecall', 'FWIoU'):
            flat[f'{k}_{alias}'] = 100 * ov[k]
        flat[f'kappa_{alias}'] = 100 * ov['Kappa']
    return flat, full
