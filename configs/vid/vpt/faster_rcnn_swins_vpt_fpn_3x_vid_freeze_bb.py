_base_ = './faster_rcnn_swint_vpt_fpn_3x_vid_freeze_bb.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
# model settings
model = dict(
    detector=dict(
        backbone=dict(
            depths=[2, 2, 18, 2],
            init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ))
)
