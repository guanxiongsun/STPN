_base_ = './yolov3_d53_mstrain-608_180e_vid.py'

# dataset settings
dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=[
            dict(
                type=dataset_type,
                ann_file=data_root + "annotations/imagenet_vid_train.json",
                img_prefix=data_root + "Data/VID",
                ref_img_sampler=None,
                pipeline=train_pipeline,
            ),
            dict(
                type=dataset_type,
                load_as_video=False,
                ann_file=data_root + "annotations/imagenet_det_30plus1cls.json",
                img_prefix=data_root + "Data/DET",
                ref_img_sampler=None,
                pipeline=train_pipeline,
            ),
        ]
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/imagenet_vid_val.json",
        img_prefix=data_root + "Data/VID",
        ref_img_sampler=None,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/imagenet_vid_val.json",
        img_prefix=data_root + "Data/VID",
        ref_img_sampler=None,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[12])      # real step is 120

# runtime settings
total_epochs = 18
checkpoint_config = dict(interval=6)
evaluation = dict(metric=["bbox"], interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)  # the real epoch is 18*5=90
