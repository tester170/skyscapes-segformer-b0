"""Swin with per-stage activation checkpointing.

mmseg's with_cp is all-or-nothing. Per block the forward FLOPs are equal across Swin stages (tokens x C^2 is
constant), but the stored activations are not: a stage-1 block keeps ~4x the elements of a stage-3 block. So
checkpointing the first blocks of the high-resolution stages buys the most memory per recomputed FLOP.
cp_blocks[i] = number of leading blocks of stage i that are checkpointed. torch.utils.checkpoint restores the RNG
state (DropPath masks) on recompute, so the math is unchanged."""
from mmseg.models.backbones import SwinTransformer
from mmseg.registry import MODELS


@MODELS.register_module()
class SwinTransformerPartialCP(SwinTransformer):

    def __init__(self, *args, cp_blocks=(0, 0, 0, 0), **kwargs):
        kwargs['with_cp'] = False
        super().__init__(*args, **kwargs)
        assert len(cp_blocks) == len(self.stages)
        for stage, n in zip(self.stages, cp_blocks):
            assert n <= len(stage.blocks)
            for block in stage.blocks[:n]:
                block.with_cp = True
        self.cp_blocks = tuple(cp_blocks)
