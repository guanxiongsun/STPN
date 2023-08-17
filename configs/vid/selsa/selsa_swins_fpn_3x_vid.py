_base_ = './selsa_swint_fpn_3x_vid.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa

model = dict(
    detector=dict(
        backbone=dict(
            depths=[2, 2, 18, 2],
            init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        )))
