_base_ = './reppoints_moment_r50_fpn_3x_vid.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(neck=dict(norm_cfg=norm_cfg), bbox_head=dict(norm_cfg=norm_cfg))
