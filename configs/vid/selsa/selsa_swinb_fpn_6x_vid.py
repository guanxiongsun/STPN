_base_ = './selsa_swint_fpn_3x_vid.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth'  # noqa
# model settings
model = dict(
    detector=dict(
        backbone=dict(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
        neck=dict(in_channels=[128, 256, 512, 1024]),))

# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[4]
)

# runtime settings
total_epochs = 6
evaluation = dict(metric=["bbox"], interval=total_epochs)
checkpoint_config = dict(interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)