_base_ = [
    "../../_base_/models/vid/faster_rcnn_r50_fpn.py",
    "../../_base_/datasets/vid/imagenet_vid_base_style.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_1x.py",
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    detector=dict(
    backbone=dict(
        _delete_=True,
        type='PromptedSwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        prompt_cfg=dict(
            num_tokens=5,
            location='prepend',
            deep=False,
            dropout=0.,
            initiation='random',
        )
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))
)

# optimizer
optimizer = dict(
    type="SGD",
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
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
