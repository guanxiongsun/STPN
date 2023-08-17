_base_ = [
    "./fcos_r50_caffe_fpn_gn-head_3x_vid_bs4_lr2e-3.py",
]

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://detectron/resnet101_caffe')))
