# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy
from mmdet.models import build_detector, build_memory
from mmdet.core import bbox2result
from ..builder import MODELS
from .base import BaseVideoDetector


@MODELS.register_module()
class CenterNetAtt(BaseVideoDetector):
    def __init__(self,
                 detector,
                 memory,
                 pretrained=None,
                 init_cfg=None,
                 frozen_modules=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CenterNetAtt, self).__init__(init_cfg)
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
        self.memory = build_memory(memory)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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
        # assert len(img) == 1, \
        #     'selsa video detector only supports 1 batch size per gpu for now.'

        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        key_x = self.detector.backbone(img)
        b, num_ref, c, h, w = ref_img.size()
        # [b*num_ref, c, h, w]
        ref_x = self.detector.backbone(ref_img.reshape(b*num_ref, c, h, w))

        # memory before or after fpn
        if self.memory.before_fpn:
            key_x = self.memory.forward_train(key_x, ref_x,
                                              gt_bboxes=gt_bboxes,
                                              ref_gt_bboxes=ref_gt_bboxes)
            if self.detector.with_neck:
                key_x = self.detector.neck(key_x)
        else:
            if self.detector.with_neck:
                key_x = self.detector.neck(key_x)
                ref_x = self.detector.neck(ref_x)
            key_x = self.memory.forward_train(key_x, ref_x,
                                              gt_bboxes=gt_bboxes,
                                              ref_gt_bboxes=ref_gt_bboxes)

        losses = self.detector.bbox_head.forward_train(key_x, img_metas, gt_bboxes,
                                                       gt_labels, gt_bboxes_ignore)

        return losses

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        """
        Extract features for `img` during testing. Backbone + Memory + Neck
        """
        frame_id = img_metas[0].get('frame_id', -1)
        assert frame_id >= 0
        # num_left_ref_imgs = img_metas[0].get('num_left_ref_imgs', -1)
        frame_stride = img_metas[0].get('frame_stride', -1)

        # test with adaptive stride
        if frame_stride < 1:
            # first frame
            # init memory with ref_frames
            if frame_id == 0:
                video = img_metas[0]["filename"].split("/")[-2]
                video_id = int(video.split("_")[-1])
                if video_id % 1000 == 0:
                    print("\n"+video)
                    self.memory.reset()
                # self.memory.reset()
                # do detection
                ref_bboxes = self.detector.simple_test(ref_img[0], ref_img_metas[0])
                # ref_bboxes += border
                border = img_metas[0]['border'][..., [2, 0, 2, 0]]
                for ref_bbox in ref_bboxes:
                    for bbox in ref_bbox:
                        # bbox[..., :-1] *= scale_factor
                        bbox[..., :-1] += border

                # write into memory
                ref_x = self.detector.backbone(ref_img[0])
                # memory before or after fpn
                if self.memory.before_fpn:
                    self.memory.write_operation(ref_x, ref_bboxes)
                else:
                    if self.detector.with_neck:
                        ref_x = self.detector.neck(ref_x)
                    self.memory.write_operation(ref_x, ref_bboxes)

            # no feats in memory
            if len(self.memory.memories[self.memory.start_level]) == 0:
                # do detection
                ref_bboxes = self.detector.simple_test(img, img_metas)

                # write into memory
                ref_x = self.detector.backbone(img)
                # memory before or after fpn
                if self.memory.before_fpn:
                    self.memory.write_operation(ref_x, ref_bboxes)
                else:
                    if self.detector.with_neck:
                        ref_x = self.detector.neck(ref_x)
                    self.memory.write_operation(ref_x, ref_bboxes)

            x = self.detector.backbone(img)
        # test with fixed stride
        else:
            raise NotImplementedError

        return x

    def simple_test(self,
                    img,
                    img_metas,
                    ref_img=None,
                    ref_img_metas=None,
                    proposals=None,
                    ref_proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (list[Tensor] | None): The list only contains one Tensor
                of shape (1, N, C, H, W) encoding input reference images.
                Typically these should be mean centered and std scaled. N
                denotes the number for reference images. There may be no
                reference images in some cases.

            ref_img_metas (list[list[list[dict]]] | None): The first and
                second list only has one element. The third list contains
                image information dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain 'filename',
                'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on
                the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.

            proposals (None | Tensor): Override rpn proposals with custom
                proposals. Use when `with_rpn` is False. Defaults to None.

            rescale (bool): If False, then returned bboxes and masks will fit
                the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The detection results.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if ref_img is not None:
            ref_img = ref_img[0]
        if ref_img_metas is not None:
            for ref_img_meta in ref_img_metas[0][0]:
                ref_img_meta['batch_input_shape'] = batch_input_shape
            ref_img_metas = ref_img_metas[0]
        x = self.extract_feats(img, img_metas, ref_img, ref_img_metas)

        # memory before or after fpn
        if self.memory.before_fpn:
            x = self.memory.forward_test(x)
            x_2b_save = x
            if self.detector.with_neck:
                x = self.detector.neck(x)
        else:
            if self.detector.with_neck:
                x = self.detector.neck(x)
            x = self.memory.forward_test(x)
            x_2b_save = x

        results_list = self.detector.bbox_head.simple_test(
            x, img_metas, rescale=rescale)
        outs = [
            bbox2result(det_bboxes, det_labels, self.detector.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        results = deepcopy(outs)

        # write into memory
        # transform from ori_img coordinate back to feat_map coordinate
        scale_factor = img_metas[0]['scale_factor']
        border = img_metas[0]['border'][..., [2, 0, 2, 0]]
        for out in outs:
            for bbox in out:
                bbox[..., :-1] *= scale_factor
                bbox[..., :-1] += border
        self.memory.write_operation(x_2b_save, outs)
        return results

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError
