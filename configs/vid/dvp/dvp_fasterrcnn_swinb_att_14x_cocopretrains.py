_base_ = [
    "./dvp_fasterrcnn_swint_att_9x_cocopretrains.py",
]
pretrained = 'work_dirs/pretrains/cascade_mask_rcnn_swin_base_patch4_window7_backbone.pth'  # noqa
# model settings
model = dict(
    embed_dims=1024,
    prompt_dims=128,
    detector=dict(
        backbone=dict(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        ),
        neck=dict(in_channels=[128, 256, 512, 1024]))
)

# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[8]
)

# runtime settings
total_epochs = 14
evaluation = dict(metric=["bbox"], interval=total_epochs)
checkpoint_config = dict(interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
