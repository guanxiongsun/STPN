_base_ = [
    "./fcos_r50_caffe_fpn_gn-head_3x_vid_bs4.py",
]

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://detectron/resnet101_caffe')))

# learning policy
lr_config = dict(
    policy="step", warmup="constant", warmup_iters=500, warmup_ratio=1.0 / 3, step=[4]
)
# runtime settings
total_epochs = 6
checkpoint_config = dict(interval=total_epochs)
evaluation = dict(metric=["bbox"], interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)