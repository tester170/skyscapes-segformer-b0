"""EncoderDecoder with a cheaper sliding-window inference.

mmseg 1.2.2 adds every window through F.pad to a full-frame tensor (C x H x W per window; 1.7 GB for 20 classes on a
5616 x 3744 frame) and runs windows one at a time. Here each window's logits are added in place into its slice
(the same additions in the same order, so the result is identical) and up to `slide_batch` windows are run per
forward pass. Window placement is copied from mmseg."""
import torch
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class FastSlideEncoderDecoder(EncoderDecoder):

    def __init__(self, *args, slide_batch=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.slide_batch = slide_batch

    def slide_inference(self, inputs, batch_img_metas):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        boxes = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y2 = min(h_idx * h_stride + h_crop, h_img)
                x2 = min(w_idx * w_stride + w_crop, w_img)
                boxes.append((max(y2 - h_crop, 0), y2, max(x2 - w_crop, 0), x2))
        preds = inputs.new_zeros((batch_size, self.out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        same_size = len({(y2 - y1, x2 - x1) for y1, y2, x1, x2 in boxes}) == 1
        step = self.slide_batch if same_size else 1
        for i in range(0, len(boxes), step):
            chunk = boxes[i:i + step]
            crops = torch.cat([inputs[:, :, y1:y2, x1:x2] for y1, y2, x1, x2 in chunk], 0)
            batch_img_metas[0]['img_shape'] = crops.shape[2:]
            metas = batch_img_metas * len(chunk)
            logits = self.encode_decode(crops, metas)
            for j, (y1, y2, x1, x2) in enumerate(chunk):
                preds[:, :, y1:y2, x1:x2] += logits[j * batch_size:(j + 1) * batch_size]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        return preds / count_mat


from mmseg.models.data_preprocessor import SegDataPreProcessor  # noqa: E402


@MODELS.register_module()
class SegDataPreProcessorU8(SegDataPreProcessor):
    """SegDataPreProcessor + non-blocking copies from pinned memory; uint8 label maps (PackSegInputsU8) are padded on
    the GPU as uint8 and cast to int64 there for training. Validation labels stay uint8 (the metric casts the masked
    pixels itself)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._non_blocking = True

    def forward(self, data, training=False):
        out = super().forward(data, training)
        if training:
            for ds in out['data_samples']:
                if ds.gt_sem_seg.data.dtype == torch.uint8:
                    ds.gt_sem_seg.data = ds.gt_sem_seg.data.long()
        return out
