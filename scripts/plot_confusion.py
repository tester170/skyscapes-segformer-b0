"""Confusion-matrix figures and full metric tables for every merge level of every chain.

Input: a val_confusion/iter_N.json written by PlateauControlHook (per-frame 20x20 matrices) or any JSON with
{"frames": {stem: matrix}}. Output dir gets:
  cm_<level>.png      row-normalised matrix (share of true-class pixels), per-class one-vs-rest kappa on the
                      diagonal, title with mIoU, overall Cohen's kappa and OA;
  curve_mIoU.png      mIoU and kappa vs number of classes for every chain, with the 0.60 target line;
  metrics.json/.csv   every per-class and overall metric of every level.
Figures are drawn in parallel processes (matplotlib is CPU-bound per figure).
Usage: python tools/plot_confusion.py <confusion.json> <out_dir> [--title-prefix TEXT]"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from skyscapes_mmseg.seg_metrics import PER_CLASS, load_chains, per_class_and_overall

MERGES = Path(__file__).resolve().parents[1] / 'splits/class_merges.json'
INK, INK_ON_DARK, MUTED = '#1f2328', '#ffffff', '#57606a'


def draw(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    key, cm, names, title, out = args
    cm = np.asarray(cm, np.float64)
    pc, ov = per_class_and_overall(cm)
    n = len(names)
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6.0, 0.62 * n + 2.5)
    fig, ax = plt.subplots(figsize=(size + 1.6, size), dpi=200)
    im = ax.imshow(rown, cmap='Blues', vmin=0, vmax=1)
    fs = 10 if n <= 10 else (8 if n <= 14 else 6.5)
    for i in range(n):
        for j in range(n):
            v = rown[i, j]
            if i != j and n > 12 and v < 0.01:
                continue
            color = INK_ON_DARK if v > 0.55 else INK
            if i == j:
                k = pc['Kappa'][i]
                ax.text(j, i - 0.12, f'{v:.2f}', ha='center', va='center', fontsize=fs, color=color)
                ax.text(j, i + 0.22, 'κ=—' if not np.isfinite(k) else f'κ={k:.2f}', ha='center', va='center',
                        fontsize=fs * 0.95, color=color, fontweight='bold')
            else:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=fs, color=color)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=fs + 1, color=MUTED)
    ax.set_yticklabels([f'{nm} (n={int(s):,})'.replace(',', ' ') for nm, s in zip(names, pc['Support'])],
                       fontsize=fs + 1, color=MUTED)
    ax.set_xlabel('Предсказанные метки', color=INK); ax.set_ylabel('Истинные метки (пикселей)', color=INK)
    ax.set_title(f'{title}\nМатрица ошибок, {n} кл.: mIoU {100 * ov["mIoU"]:.2f}%  ·  Kappa {ov["Kappa"]:.4f}  ·  '
                 f'OA {100 * ov["OA"]:.2f}%  ·  mDice {100 * ov["mDice"]:.2f}%', color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n), minor=True); ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5); ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED)
    cb.set_label('доля пикселей истинного класса', color=MUTED)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    return key


def draw_curves(curves, title, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'A': '#2a6fdb', 'B': '#e07b00', 'C': '#1a9e77'}  # fixed order per chain identity
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200, sharex=True)
    for ax, key, label in ((axes[0], 'mIoU', 'mIoU, %'), (axes[1], 'Kappa', 'Kappa')):
        for c, pts in curves.items():
            x = [p['n'] for p in pts]; y = [p[key] for p in pts]
            ax.plot(x, y, color=colors.get(c, MUTED), lw=2, marker='o', ms=5, label=f'цепочка {c}')
        if key == 'mIoU':
            ax.axhline(60, color=MUTED, lw=1, ls='--'); ax.text(20, 60.5, 'цель 60%', ha='right', color=MUTED, fontsize=9)
        ax.set_xlabel('число классов'); ax.set_ylabel(label); ax.invert_xaxis()
        ax.grid(color='#d0d7de', lw=0.6); [s.set_visible(False) for s in ax.spines.values()]
        ax.legend(frameon=False)
    fig.suptitle(title, color=INK); fig.tight_layout(); fig.savefig(out, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('confusion'); ap.add_argument('out'); ap.add_argument('--title-prefix', default='')
    ap.add_argument('--merges', default=str(MERGES), help='merge-chain JSON (class and group names come from it)')
    ap.add_argument('--chains', nargs='*', help='only these chains')
    ap.add_argument('--chain-title', default='цепочка', help='word used before the chain name in titles; none = omit the chain label')
    a = ap.parse_args()
    src = json.loads(Path(a.confusion).read_text())
    cm = np.sum([np.asarray(m) for m in src['frames'].values()], 0)
    chains, levels, names0 = load_chains(a.merges)
    if a.chains:
        chains = {c: v for c, v in chains.items() if c in a.chains}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs, table, curves = [], {}, {}
    seen = set()
    for c, steps in chains.items():
        curves[c] = []
        for n, G, names in steps:
            key = 'L0_20' if n == len(names0) else f'{c}{n}'
            # probability-sum merged matrix saved at validation when available, else hard mapping of the argmax
            lv = (src.get('levels_cm') or {}).get(f'{c}{n}')
            m = G.T @ cm @ G if lv is None else np.asarray(lv)
            pc, ov = per_class_and_overall(m)
            curves[c].append(dict(n=n, mIoU=100 * ov['mIoU'], Kappa=ov['Kappa']))
            if key in seen:
                continue
            seen.add(key)
            alias = [al for al, (cc, nn) in levels.items() if cc == c and nn == n]
            title = f'{a.title_prefix} {("" if a.chain_title == "none" else (a.chain_title + " " + c)) if n < len(names0) else "исходные классы"}' \
                    f'{" (" + alias[0] + ")" if alias else ""}'.strip()
            table[key] = dict(names=names, overall={k: float(v) for k, v in ov.items()},
                              per_class={k: [None if not np.isfinite(x) else float(x) for x in pc[k]] for k in pc})
            jobs.append((key, m.tolist(), names, title, str(out / f'cm_{key}.png')))
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(draw, jobs))
    draw_curves(curves, f'{a.title_prefix} mIoU и Kappa при слиянии классов'.strip(), out / 'curve_mIoU.png')
    (out / 'metrics.json').write_text(json.dumps(dict(source=a.confusion, levels=table, curves=curves), indent=1,
                                                ensure_ascii=False))
    with open(out / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['level', 'class'] + list(PER_CLASS) + ['Support'])
        for key, t in table.items():
            for i, nm in enumerate(t['names']):
                w.writerow([key, nm] + [t['per_class'][k][i] for k in PER_CLASS] + [t['per_class']['Support'][i]])
            w.writerow([key, 'OVERALL'] + [t['overall'].get('m' + k) for k in PER_CLASS] + [None])
    print(f'{len(jobs)} matrices + curves -> {out}')


if __name__ == '__main__':
    main()
