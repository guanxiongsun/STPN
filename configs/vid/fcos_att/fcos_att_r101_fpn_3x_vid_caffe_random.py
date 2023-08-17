_base_ = [
    './fcos_att_r50_fpn_3x_vid_caffe_random.py'
]

model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://detectron/resnet101_caffe'))))
