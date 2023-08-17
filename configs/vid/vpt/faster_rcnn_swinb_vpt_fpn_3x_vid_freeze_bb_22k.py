_base_ = './faster_rcnn_swint_vpt_fpn_3x_vid_freeze_bb.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa
# model settings
model = dict(
    detector=dict(
        backbone=dict(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        ),
        neck=dict(in_channels=[128, 256, 512, 1024]))
)
