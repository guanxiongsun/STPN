_base_ = [
    "./dvp_selsa_swint_att_9x_cocopretrains.py"
]
pretrained = 'work_dirs/pretrains/cascade_mask_rcnn_swin_base_patch4_window7_backbone.pth'  # noqa

# model settings
model = dict(
    predictor='avg',
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
