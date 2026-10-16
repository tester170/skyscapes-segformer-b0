"""Download the LoveDA-pretrained MiT-B0 encoder used to initialise the experiment and convert it to MMSegmentation.

Source: Hugging Face Pranilllllll/geonusaf-segformer-b0-loveda-both-v3-slide, file best.pt (Hugging Face SegFormer
key names). Only the encoder is kept (the source head is not used); keys are mapped to MMSeg MixVisionTransformer
exactly as tools/model_converters/mit2mmseg.py does (K and V are concatenated after Q into attn.in_proj).
The result is identical, tensor by tensor, to checkpoints/mit_b0_loveda_encoder.pth of the experiment.
Usage: python scripts/get_encoder.py [--out checkpoints/mit_b0_loveda_encoder.pth] [--source best.pt]"""
import argparse
import hashlib
import re
import urllib.request
from collections import OrderedDict
from pathlib import Path

import torch

URL = 'https://huggingface.co/Pranilllllll/geonusaf-segformer-b0-loveda-both-v3-slide/resolve/main/best.pt'
SOURCE_SHA256 = '8345b59487a35c1cb68260e976452eb9e2a968c9d52eaf4758c924f5bd057809'
RESULT_DIGEST = '019760d019728a2c18fc4555200c17687c8cc0b60457782b0732b66112df2916'   # sha256 over the converted tensors (see digest()), filled in by make_colab_bundle.py
BUNDLE = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(state):
    """Order-independent fingerprint of a state dict: names, dtypes, shapes and raw bytes of every tensor."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].contiguous().cpu()
        h.update(f'{k}|{t.dtype}|{tuple(t.shape)}|'.encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b'')
    return h.hexdigest()


def convert(hf):
    """Hugging Face SegFormer encoder keys -> MMSeg MixVisionTransformer keys (backbone.layers.*)."""
    out = OrderedDict()
    for key, v in hf.items():
        if not key.startswith('encoder.encoder.'):
            continue                                            # source decode head is not used
        k = key[len('encoder.encoder.'):]
        m = re.fullmatch(r'patch_embeddings\.(\d+)\.(proj|layer_norm)\.(weight|bias)', k)
        if m:
            s, layer, p = m.groups()
            out[f'layers.{s}.0.{"projection" if layer == "proj" else "norm"}.{p}'] = v
            continue
        m = re.fullmatch(r'layer_norm\.(\d+)\.(weight|bias)', k)
        if m:
            out[f'layers.{m[1]}.2.{m[2]}'] = v
            continue
        s, b, tail = re.fullmatch(r'block\.(\d+)\.(\d+)\.(.+)', k).groups()
        pre = f'layers.{s}.1.{b}.'
        if tail.startswith(('attention.self.key.', 'attention.self.value.')):
            continue                                            # merged into in_proj together with the query
        if tail.startswith('attention.self.query.'):
            p = tail.rsplit('.', 1)[1]
            base = key.rsplit('attention.self.query.', 1)[0] + 'attention.self.'
            out[pre + f'attn.attn.in_proj_{p}'] = torch.cat([v, hf[base + 'key.' + p], hf[base + 'value.' + p]], 0)
            continue
        table = {'layer_norm_1.': 'norm1.', 'layer_norm_2.': 'norm2.', 'attention.self.sr.': 'attn.sr.',
                 'attention.self.layer_norm.': 'attn.norm.', 'attention.output.dense.': 'attn.attn.out_proj.',
                 'mlp.dense1.': 'ffn.layers.0.', 'mlp.dwconv.dwconv.': 'ffn.layers.1.', 'mlp.dense2.': 'ffn.layers.4.'}
        src = next(x for x in table if tail.startswith(x))
        new = pre + table[src] + tail[len(src):]
        if src in ('mlp.dense1.', 'mlp.dense2.') and new.endswith('weight'):
            v = v.reshape(*v.shape, 1, 1)                       # linear -> 1x1 conv
        out[new] = v
    return {'backbone.' + k: t for k, t in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(BUNDLE / 'checkpoints' / 'mit_b0_loveda_encoder.pth'))
    ap.add_argument('--source', default=None, help='local copy of best.pt (otherwise downloaded)')
    a = ap.parse_args()
    out = Path(a.out)
    src = Path(a.source) if a.source else out.with_name('loveda_best_source.pt')
    if not src.exists() or sha256(src) != SOURCE_SHA256:
        print(f'downloading {URL}', flush=True)
        urllib.request.urlretrieve(URL, src)
    assert sha256(src) == SOURCE_SHA256, f'{src}: checksum mismatch, the source file changed'
    state = convert(torch.load(src, map_location='cpu', weights_only=True)['model'])
    assert len(state) == 160, len(state)
    d = digest(state)
    assert RESULT_DIGEST in ('SET_BY_BUNDLE', d), f'converted encoder differs from the experiment ({d})'
    torch.save(dict(state_dict=state, meta=dict(source='Pranilllllll/geonusaf-segformer-b0-loveda-both-v3-slide',
                                                source_sha256=SOURCE_SHA256, digest=d)), out)
    if not a.source:
        src.unlink()
    print(f'encoder ready: {out} ({len(state)} tensors, digest {d[:16]})', flush=True)


if __name__ == '__main__':
    main()
