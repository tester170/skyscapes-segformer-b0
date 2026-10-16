# SegFormer-B0 for SkyScapes-Dense, MMSegmentation 1.2.2 (resolved, self-contained). Data paths, merge file and
# initial weights are set by scripts/*.py at run time.
crop_size = (
    1024,
    1024,
)
custom_hooks = [
    dict(type='VramLogHook'),
    dict(
        cooldown=2,
        ema_alpha=0.5,
        factor=0.5,
        lr_patience=5,
        max_reductions=4,
        min_delta=0.1,
        monitor='mIoU',
        start_iter=1500,
        stop_patience=12,
        type='PlateauControlHook'),
    dict(interval_s=20.0, type='ConsoleProgressHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'skyscapes_mmseg',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        1024,
        1024,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessorU8')
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=30,
        max_keep_ckpts=1,
        rule='greater',
        save_best='mIoU',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fold = 0
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True)
max_epochs = 600
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=32,
        in_channels=3,
        init_cfg=None,
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            2,
            2,
            2,
            2,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            1024,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessorU8'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=20,
        type='SegformerHead'),
    pretrained=None,
    slide_batch=1,
    test_cfg=dict(crop_size=(
        1024,
        1024,
    ), mode='slide', stride=(
        768,
        768,
    )),
    train_cfg=dict(),
    type='FastSlideEncoderDecoder')
monitor = 'mIoU'
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
]
randomness = dict(seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        cache_root='SET_BY_SCRIPTS',
        fold=0,
        folds_file='SET_BY_SCRIPTS',
        pipeline=[
            dict(type='LoadCachedFrame'),
            dict(type='PackSegInputsU8'),
        ],
        split='val',
        type='SkyScapesFrames'),
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(merges_file='SET_BY_SCRIPTS', type='ConfusionIoUMetric')
train_cfg = dict(max_epochs=600, type='EpochBasedTrainLoop', val_interval=10)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        cache_root='SET_BY_SCRIPTS',
        crop_size=(
            1024,
            1024,
        ),
        fold=0,
        folds_file='SET_BY_SCRIPTS',
        pipeline=[
            dict(
                brightness_delta=16,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=5,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='FastPhotoMetricDistortion'),
            dict(type='RandomDihedral'),
            dict(type='PackSegInputsU8'),
        ],
        ratio_range=(
            0.75,
            1.5,
        ),
        rcs_fraction=0.25,
        rcs_temperature=0.01,
        type='SkyScapesTiles'),
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        brightness_delta=16,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=5,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='FastPhotoMetricDistortion'),
    dict(type='RandomDihedral'),
    dict(type='PackSegInputsU8'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        cache_root='SET_BY_SCRIPTS',
        fold=0,
        folds_file='SET_BY_SCRIPTS',
        pipeline=[
            dict(type='LoadCachedFrame'),
            dict(type='PackSegInputsU8'),
        ],
        split='val',
        type='SkyScapesFrames'),
    num_workers=1,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(merges_file='SET_BY_SCRIPTS', type='ConfusionIoUMetric')
val_interval = 10
val_pipeline = [
    dict(type='LoadCachedFrame'),
    dict(type='PackSegInputsU8'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
warmup_iters = 1500
