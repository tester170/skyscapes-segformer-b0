"""Runtime patch for mmseg 1.2.2 MiT (SegFormer) attention.

EfficientMultiheadAttention.forward calls nn.MultiheadAttention with the default need_weights=True, so PyTorch takes
the explicit path: it materialises the full attention matrix (B x heads x N_q x N_kv, fp32 softmax under autocast),
keeps it for backward and averages it into weights that mmseg then discards. With need_weights=False PyTorch >= 2.0
calls scaled_dot_product_attention (memory-efficient / flash kernels) and returns the same attention output.
mmseg's own legacy_forward already passes need_weights=False. attn_drop_rate is 0 in all SegFormer configs, so
dropout does not differ between the paths. Equivalence is checked numerically in tools/e1_check_attention.py.
"""
from mmseg.models.backbones.mit import EfficientMultiheadAttention
from mmseg.models.utils import nchw_to_nlc, nlc_to_nchw


def _forward_no_weights(self, x, hw_shape, identity=None):
    x_q = x
    if self.sr_ratio > 1:
        x_kv = nlc_to_nchw(x, hw_shape)
        x_kv = self.sr(x_kv)
        x_kv = nchw_to_nlc(x_kv)
        x_kv = self.norm(x_kv)
    else:
        x_kv = x
    if identity is None:
        identity = x_q
    if self.batch_first:
        x_q = x_q.transpose(0, 1)
        x_kv = x_kv.transpose(0, 1)
    out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]
    if self.batch_first:
        out = out.transpose(0, 1)
    return identity + self.dropout_layer(self.proj_drop(out))


ORIGINAL_FORWARD = EfficientMultiheadAttention.forward
EfficientMultiheadAttention.forward = _forward_no_weights


# Training-set confusion matrix at no extra resize: BaseDecodeHead.loss_by_feat already upsamples the logits to the
# label size for `accuracy`; the wrapper adds one argmax + bincount on the GPU (no host sync) for the head whose
# `track_train_confusion` flag is set (the main head; auxiliary heads are not counted). Heads that override
# loss_by_feat without calling this one (e.g. Mask2Former) are not tracked.
import torch  # noqa: E402
import mmseg.models.decode_heads.decode_head as _decode_head_module  # noqa: E402
from mmseg.models.decode_heads.decode_head import BaseDecodeHead  # noqa: E402


class _TrainConfusion:
    active = False
    k = None
    ignore_index = 255
    cm = None

    def reset(self):
        self.cm = None

    def take(self):
        cm, self.cm = self.cm, None
        return None if cm is None else cm.reshape(self.k, self.k).cpu().numpy()


TRAIN_CONFUSION = _TrainConfusion()
_ORIGINAL_ACCURACY = _decode_head_module.accuracy
_ORIGINAL_LOSS_BY_FEAT = BaseDecodeHead.loss_by_feat


def _accuracy_with_confusion(pred, target, *args, **kwargs):
    tc = TRAIN_CONFUSION
    if tc.active and pred.dim() == 4:
        with torch.no_grad():
            m = target != tc.ignore_index
            bc = torch.bincount(target[m] * tc.k + pred.argmax(1)[m], minlength=tc.k * tc.k)
            tc.cm = bc if tc.cm is None else tc.cm + bc
    return _ORIGINAL_ACCURACY(pred, target, *args, **kwargs)


def _loss_by_feat_tracking(self, seg_logits, batch_data_samples):
    tc = TRAIN_CONFUSION
    tc.active = bool(getattr(self, 'track_train_confusion', False))
    tc.k, tc.ignore_index = self.out_channels, self.ignore_index
    try:
        return _ORIGINAL_LOSS_BY_FEAT(self, seg_logits, batch_data_samples)
    finally:
        tc.active = False


_decode_head_module.accuracy = _accuracy_with_confusion
BaseDecodeHead.loss_by_feat = _loss_by_feat_tracking
