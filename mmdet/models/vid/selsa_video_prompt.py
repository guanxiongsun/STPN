# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
from addict import Dict
from mmdet.models import build_detector

from ..builder import MODELS
from .base import BaseVideoDetector
from ..predictors import AveragePredictor, AttentionPredictor


@MODELS.register_module()
class SELSAVideoPrompt(BaseVideoDetector):
    """Sequence Level Semantics Aggregation for Video Object Detection.

    This video object detector is the implementation of `SELSA
    <https://arxiv.org/abs/1907.06390>`_.
    """

    def __init__(self,
                 detector,
                 pretrained=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None,
                 predictor='att',
                 extra_frames=False,
                 embed_dims=768,
                 num_prompts=5,
                 prompt_dims=96,
                 ):
        super(SELSAVideoPrompt, self).__init__(init_cfg)
        if isinstance(pretrained, dict):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            detector_pretrain = pretrained.get('detector', None)
            if detector_pretrain:
                detector.init_cfg = dict(
                    type='Pretrained', checkpoint=detector_pretrain)
            else:
                detector.init_cfg = None
        self.detector = build_detector(detector)
        assert hasattr(self.detector, 'roi_head'), \
            'selsa video detector only supports two stage detector'
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # create prompt predict network
        print("The predictor is {}".format(predictor))
        if predictor == 'att':
            self.prompt_predictor = AttentionPredictor(embed_dims,
                                                       num_prompts=num_prompts,
                                                       prompt_dims=prompt_dims)
        elif predictor == 'avg':
            self.prompt_predictor = AveragePredictor(embed_dims,
                                                     num_prompts=num_prompts,
                                                     prompt_dims=prompt_dims,
                                                     )
        else:
            raise ValueError

        self.extra_frames = extra_frames

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_instance_ids=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      ref_gt_instance_ids=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_proposals=None,
                      **kwargs):
        assert len(img) == 1, \
            'selsa video detector only supports 1 batch size per gpu for now.'

        if self.extra_frames:
            # ref_x [4, C, H, W]
            prompt_ref = ref_img[0][:2]
            selsa_ref = ref_img[0][2:]
            with torch.no_grad():
                prompt_ref = self.detector.backbone(prompt_ref)
            prompt = self.prompt_predictor(prompt_ref[-1])

            all_imgs = torch.cat([img, selsa_ref])
            all_x = self.detector.backbone(all_imgs, prompt)
            # ref pass fpn
            if self.detector.with_neck:
                all_x = self.detector.neck(all_x)

            x = []
            ref_x = []
            for i in range(len(all_x)):
                x.append(all_x[i][[0]])
                ref_x.append(all_x[i][1:])
        else:
            # [B, C, H, W]
            # ref w.o. prompt
            ref_x = self.detector.backbone(ref_img[0])

            # [num_prompt, C]
            # key w. prompt
            prompt = self.prompt_predictor(ref_x[-1])
            x = self.detector.backbone(img, prompt)
            if self.detector.with_neck:
                x = self.detector.neck(x)

            # ref pass fpn
            if self.detector.with_neck:
                ref_x = self.detector.neck(ref_x)

        losses = dict()

        # RPN forward and loss
        if self.detector.with_rpn:
            proposal_cfg = self.detector.train_cfg.get(
                'rpn_proposal', self.detector.test_cfg.rpn)
            rpn_losses, proposal_list = self.detector.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)

            if self.extra_frames:
                ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                    ref_x, ref_img_metas[0][2:])
            else:
                ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                    ref_x, ref_img_metas[0])
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        roi_losses = self.detector.roi_head.forward_train(
            x, ref_x, img_metas, proposal_list, ref_proposals_list, gt_bboxes,
            gt_labels, gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)

        return losses

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        """Extract features for `img` during testing.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (Tensor | None): of shape (1, N, C, H, W) encoding input
                reference images. Typically these should be mean centered and
                std scaled. N denotes the number of reference images. There
                may be no reference images in some cases.

            ref_img_metas (list[list[dict]] | None): The first list only has
                one element. The second list contains image information dict
                where each dict has: 'img_shape', 'scale_factor', 'flip', and
                may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

        Returns:
            tuple(x, img_metas, ref_x, ref_img_metas): x is the multi level
                feature maps of `img`, ref_x is the multi level feature maps
                of `ref_img`.
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', -1)
        frame_stride = img_metas[0].get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            if frame_id == 0:
                self.memo = Dict()
                if self.extra_frames:
                    self.memo.img_metas = ref_img_metas[0][::2]
                    # ref for prompt
                    prompt_ref = ref_img[0][1::2]
                    selsa_ref = ref_img[0][::2]

                    prompt_ref = self.detector.backbone(prompt_ref)
                    self.prompt = self.prompt_predictor(prompt_ref[-1])

                    ref_x = self.detector.backbone(selsa_ref, self.prompt)
                    if self.detector.with_neck:
                        ref_x = self.detector.neck(ref_x)

                else:
                    self.memo.img_metas = ref_img_metas[0]
                    # extract feature maps w.o. prompt
                    # [B, C, H, W]
                    ref_x = self.detector.backbone(ref_img[0])
                    # predict prompts
                    self.prompt = self.prompt_predictor(ref_x[-1])
                    # ref pass fpn
                    if self.detector.with_neck:
                        ref_x = self.detector.neck(ref_x)

                # 'tuple' object (e.g. the output of FPN) does not support
                # item assignment
                self.memo.feats = []
                for i in range(len(ref_x)):
                    self.memo.feats.append(ref_x[i])

            # use prompts
            x = self.detector.backbone(img, self.prompt)
            if self.detector.with_neck:
                x = self.detector.neck(x)

            ref_x = self.memo.feats.copy()
            for i in range(len(x)):
                ref_x[i] = torch.cat((ref_x[i], x[i]), dim=0)
            ref_img_metas = self.memo.img_metas.copy()
            ref_img_metas.extend(img_metas)

        return x, img_metas, ref_x, ref_img_metas

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    ref_proposals=None,
                    rescale=False):
        if ref_img is not None:
            ref_img = ref_img[0]
        if ref_img_metas is not None:
            ref_img_metas = ref_img_metas[0]
        x, img_metas, ref_x, ref_img_metas = self.extract_feats(
            img, img_metas, ref_img, ref_img_metas)

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(
                x, img_metas)
            ref_proposals_list = self.detector.rpn_head.simple_test_rpn(
                ref_x, ref_img_metas)
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        outs = self.detector.roi_head.simple_test(
            x,
            ref_x,
            proposal_list,
            ref_proposals_list,
            img_metas,
            rescale=rescale)

        # results = dict()
        # results['det_bboxes'] = outs[0]
        # if len(outs) == 2:
        #     results['det_masks'] = outs[1]

        results = outs
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
