_base_ = [
    '../../_base_/datasets/vid/imagenet_vid_fgfa_style.py',
    '../../_base_/default_runtime.py',
    "../../_base_/schedules/schedule_1x.py",
]

# model settings
model = dict(
    type='FCOSAtt',
    detector=dict(
        type='FCOS',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron/resnet50_caffe')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',  # use P5
            num_outs=5,
            relu_before_extra_convs=True),
        bbox_head=dict(
            type='FCOSHead',
            num_classes=30,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            strides=[8, 16, 32, 64, 128],
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='IoULoss', loss_weight=1.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)),
    memory=dict(
        type='MPN',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        before_fpn=True,
        start_level=1,
    ),
)

# dataset settings
data = dict(
    val=dict(
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')),
    test=dict(
        ref_img_sampler=dict(
            _delete_=True,
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride')))

# optimizer
optimizer = dict(
    type="SGD",
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[2]
)
# runtime settings
total_epochs = 3
checkpoint_config = dict(interval=3)
evaluation = dict(metric=["bbox"], interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
