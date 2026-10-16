"""Numba version of mmseg PhotoMetricDistortion with bit-identical output and identical np.random consumption.

mmseg makes a float32 temporary, a clip temporary and a uint8 copy for every brightness/contrast/saturation step and
an int64 temporary for hue. Here random numbers are drawn exactly as mmseg draws them, then each step runs as one
fused Numba pass in float32 without FMA contraction (fastmath off), keeping every intermediate uint8 rounding.
HSV conversions stay with cv2 (as in mmcv.bgr2hsv / hsv2bgr), because their integer tables define the result.
"""
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from numba import njit
from numpy import random


@njit(cache=True, nogil=True)
def _affine2_u8(x, a1, b1, use1, a2, b2, use2, out):
    """out = q(q(x*a1+b1)*a2+b2) with q = clip to [0,255] and truncate to uint8; each stage optional."""
    for i in range(x.size):
        v = x[i]
        if use1:
            f = np.float32(v) * a1 + b1
            f = min(max(f, np.float32(0.0)), np.float32(255.0))
            v = np.uint8(f)
        if use2:
            f = np.float32(v) * a2 + b2
            f = min(max(f, np.float32(0.0)), np.float32(255.0))
            v = np.uint8(f)
        out[i] = v


@njit(cache=True, nogil=True)
def _sat_u8(hsv, a):
    for i in range(hsv.shape[0]):
        f = np.float32(hsv[i, 1]) * a
        f = min(max(f, np.float32(0.0)), np.float32(255.0))
        hsv[i, 1] = np.uint8(f)


@njit(cache=True, nogil=True)
def _hue_u8(hsv, d):
    for i in range(hsv.shape[0]):
        hsv[i, 0] = np.uint8((np.int64(hsv[i, 0]) + d) % 180)


@TRANSFORMS.register_module()
class FastPhotoMetricDistortion(BaseTransform):

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def _draw(self):
        """Same calls in the same order as mmseg PhotoMetricDistortion.transform."""
        beta = random.uniform(-self.brightness_delta, self.brightness_delta) if random.randint(2) else None
        mode = random.randint(2)
        c1 = random.uniform(self.contrast_lower, self.contrast_upper) if mode == 1 and random.randint(2) else None
        sat = random.uniform(self.saturation_lower, self.saturation_upper) if random.randint(2) else None
        hue = random.randint(-self.hue_delta, self.hue_delta) if random.randint(2) else None
        c0 = random.uniform(self.contrast_lower, self.contrast_upper) if mode == 0 and random.randint(2) else None
        return beta, c1, sat, hue, c0

    @staticmethod
    def _affine(img, a1, b1, use1, a2=1.0, b2=0.0, use2=False):
        out = np.empty(img.shape, np.uint8)
        src = np.ascontiguousarray(img)
        _affine2_u8(src.reshape(-1), np.float32(a1), np.float32(b1), use1,
                    np.float32(a2), np.float32(b2), use2, out.reshape(-1))
        return out

    def transform(self, results):
        img = results['img']
        beta, c1, sat, hue, c0 = self._draw()
        if beta is not None or c1 is not None:
            img = self._affine(img, 1.0, 0.0 if beta is None else beta, beta is not None,
                               1.0 if c1 is None else c1, 0.0, c1 is not None)
        if sat is not None:
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV)
            _sat_u8(hsv.reshape(-1, 3), np.float32(sat))
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if hue is not None:
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV)
            _hue_u8(hsv.reshape(-1, 3), np.int64(hue))
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if c0 is not None:
            img = self._affine(img, c0, 0.0, True)
        results['img'] = img
        return results


@njit(cache=True, nogil=True)
def _add_clip_u8(x, noise, out):
    for i in range(x.size):
        v = np.int32(x[i]) + np.int32(noise[i])
        out[i] = np.uint8(min(max(v, 0), 255))


@TRANSFORMS.register_module()
class RandomBlurNoise(BaseTransform):
    """Sensor/optics variation: Gaussian blur (sigma ~ U(blur_sigma) px, prob blur_prob) and additive Gaussian noise
    (sigma ~ U(noise_sigma) DN, prob noise_prob). Image only; labels untouched. Randomness is driven by np.random
    (cv2's RNG is reseeded from it), so samples are reproducible under mmengine worker seeding."""

    def __init__(self, blur_prob=0.2, blur_sigma=(0.3, 1.0), noise_prob=0.2, noise_sigma=(1.0, 5.0)):
        self.blur_prob, self.blur_sigma = blur_prob, tuple(blur_sigma)
        self.noise_prob, self.noise_sigma = noise_prob, tuple(noise_sigma)

    def transform(self, results):
        img = results['img']
        if random.rand() < self.blur_prob:
            img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), random.uniform(*self.blur_sigma))
        if random.rand() < self.noise_prob:
            sigma = random.uniform(*self.noise_sigma)
            cv2.setRNGSeed(int(random.randint(2 ** 31 - 1)))
            noise = np.empty(img.shape, np.int16)
            cv2.randn(noise, 0, sigma)
            src = np.ascontiguousarray(img)
            out = np.empty_like(src)
            _add_clip_u8(src.reshape(-1), noise.reshape(-1), out.reshape(-1))
            img = out
        results['img'] = img
        return results
