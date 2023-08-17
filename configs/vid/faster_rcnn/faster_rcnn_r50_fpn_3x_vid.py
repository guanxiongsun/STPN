_base_ = [
    "../../_base_/models/vid/faster_rcnn_r50_fpn.py",
    "../../_base_/datasets/vid/imagenet_vid_base_style.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_1x.py",
]

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
