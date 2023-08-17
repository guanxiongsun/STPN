# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule


class AveragePredictor(BaseModule):
    def __init__(self, in_channels, init_cfg=None,
                 num_prompts=5, prompt_dims=96
                 ):
        super(AveragePredictor, self).__init__(init_cfg)
        self.num_prompts = num_prompts
        self.reduction = nn.Linear(in_channels, prompt_dims)

    @staticmethod
    def get_topk(x, k=100):
        # x [B, N, C]
        result = []
        for feat in x:
            l1 = feat.norm(1, dim=-1)
            _, inds = l1.topk(k)
            result.append(feat[inds])

        return torch.stack(result).mean(dim=0)

    def forward(self, ref_x):
        # [B*K, C]
        B, C, H, W = ref_x.shape
        ref_x = ref_x.view(B, C, -1).permute(0, 2, 1)
        ref_x = self.get_topk(ref_x, k=self.num_prompts)

        # [n, c] -> [n, emb]
        ref_x = self.reduction(ref_x)

        return ref_x

    def forward_seq(self, ref_x):
        # [B, L, C]
        ref_x = self.get_topk(ref_x, k=self.num_prompts)

        # [n, c] -> [n, emb]
        ref_x = self.reduction(ref_x)

        return ref_x
