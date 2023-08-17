_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_1x.py",
]

model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=2048,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=30,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))


# dataset settings
dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# from imagenet_vid_base_style
# test_pipeline = [
#     dict(
#         type="MultiScaleFlipAug",
#         img_scale=(1000, 600),
#         flip=False,
#         transforms=[
#             dict(type="Resize", keep_ratio=True),
#             dict(type="RandomFlip"),
#             dict(type="Normalize", **img_norm_cfg),
#             dict(type="Pad", size_divisor=16),
#             dict(type="ImageToTensor", keys=["img"]),
#             dict(type="VideoCollect", keys=["img"]),
#         ],
#     ),
# ]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

# Use RepeatDataset to speed up training
# data = dict(
#     train=dict(
#         _delete_=True,
#         type='RepeatDataset',
#         times=5,
#         dataset=dict(
#             type=dataset_type,
#             ann_file=data_root + 'annotations/instances_train2017.json',
#             img_prefix=data_root + 'train2017/',
#             pipeline=train_pipeline)),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))

data = dict(
    samples_per_gpu=8,
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

# optimizer
optimizer = dict(
    type="SGD",
    lr=1e-4,
    momentum=0.9,
    weight_decay=0.0001,
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[10])  # the real step is [10*5]

# runtime settings
total_epochs = 16
checkpoint_config = dict(interval=10)
evaluation = dict(metric=["bbox"], interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)  # the real epoch is 16*5=80
