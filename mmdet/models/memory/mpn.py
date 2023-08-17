# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .memory_bank import MemoryBank


from ..builder import NECKS


@NECKS.register_module()
class MPN(BaseModule):
    r"""Memory Pyramid Network.
    Args:
        in_channels (List[int]): Number of input channels per scale.
    """

    def __init__(self, in_channels,
                 strides,
                 before_fpn,
                 start_level,
                 pixel_sampling_train='bbox',
                 ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.strides = strides
        self.before_fpn = before_fpn
        self.start_level = start_level
        self.pixel_sampling_train = pixel_sampling_train

        # add memory modules for every level
        self.memories = nn.ModuleList()
        for lvl, _in_channels in enumerate(self.in_channels):
            if lvl < self.start_level:
                _memory = nn.Identity()
            else:
                _memory = MemoryBank(_in_channels)
            self.memories.append(_memory)

    @staticmethod
    def get_feats_randomly_single_level(x):
        """
        save pixels within detected boxes into memory
        :param x: [N, C, H, W]
        :return
        """
        n, c, _, w = x.size()
        feats_list = []
        for i, _x in enumerate(x):
            # [C, H, W] -> [H*W, C]
            _x = _x.view(c, -1).permute(1, 0).contiguous()
            MAX_NUM_PER_IMG = 2000
            if len(_x) < MAX_NUM_PER_IMG:
                feats_list.append(_x)
            else:
                # randomly select 2000
                inds = np.arange(len(_x))
                np.random.shuffle(inds)
                feats_list.append(_x[inds[:MAX_NUM_PER_IMG]])

        return torch.cat(feats_list, dim=0)

    def get_ref_feats_from_gtbboxes_single_level_train(self, x, gt_bboxes, stride):
        """
        save pixels within detected boxes into memory
        :param x: [N, C, H, W]
        :param gt_bboxes: [n, 5(ind, x, y, x, y)], e.g.,
            tensor([[  0.0000, 329.8406, 247.6415, 591.7313, 428.7736],
                    [  1.0000, 305.7750, 247.6415, 580.4062, 431.6038]], device='cuda:0')
        :param stride: stride of the current level
        :return
        """
        n, c, _, w = x.size()
        ref_obj_list = []
        for i, _x in enumerate(x):
            # [C, H, W] -> [H*W, C]
            _x = _x.view(c, -1).permute(1, 0).contiguous()

            # get bboxes of this image
            ind = gt_bboxes[:, 0] == i
            _bboxes = gt_bboxes[ind]     # [n0, 4]
            # no object
            if len(_bboxes) == 0:
                continue
            # have objects
            else:
                _bboxes = _bboxes[:, 1:]
                boxes = torch.div(_bboxes, stride).int()
                for box in boxes:
                    # 1. map pixels in box to new index on x_box [H*W, C]
                    # box [x1, y1, x2, y2] -> [ind_1, ind_2, ind_3, ... ]
                    inds = sorted(self.box_to_inds_list(box, w))
                    inds = np.asarray(inds)
                    # obj too small
                    if len(inds) == 0:
                        obj_irr_inds = self.get_obj_irr_inds_topk(_x, 10)
                        ref_obj_list.append(_x[obj_irr_inds])
                    # save part obj
                    else:
                        PIXEL_NUM = 300
                        if len(inds) > PIXEL_NUM:
                            inds = np.random.choice(inds, PIXEL_NUM, replace=False)
                        inds = np.clip(inds, 0, len(_x)-1)
                        ref_obj_list.append(_x[inds])

        # no obj
        if len(ref_obj_list) == 0:
            # obj irr pixels
            IRR_PIXEL_NUM = 50
            obj_irr_inds = self.get_obj_irr_inds_topk(_x, IRR_PIXEL_NUM)
            return _x[obj_irr_inds]

        # max num of feats is set to 1000
        ref_all = torch.cat(ref_obj_list, dim=0)
        return ref_all[:1000]

    @staticmethod
    @torch.no_grad()
    def box_to_inds_list(box, w):
        inds = []
        for x_i in range(box[0], box[2] + 1):
            for y_j in range(box[1], box[3] + 1):
                inds.append(int(x_i + y_j * w))
        return inds

    @staticmethod
    @torch.no_grad()
    def get_obj_irr_inds_topk(x, k=50):
        """
        get top k object irrelevant features
        :param x: [n, c]
        :param k: number of pixels
        :return: [m, c]
        """
        n, c = x.size()
        k = min(n, k)
        l2_norm = x.pow(2).sum(dim=1).sqrt() / np.sqrt(c)
        _, inds = l2_norm.topk(k)
        return inds

    def prepare_memory_train(self, ref_x, ref_gt_bboxes):
        ref_feats_all = []
        for lvl in range(len(ref_x)):
            if lvl < self.start_level:
                ref_feats_all.append(None)
                continue

            _ref_x = ref_x[lvl]
            if self.pixel_sampling_train == 'random':
                _ref_feats = self.get_feats_randomly_single_level(_ref_x)
            elif self.pixel_sampling_train == 'bbox':
                _ref_feats = self.get_ref_feats_from_gtbboxes_single_level_train(
                    _ref_x.clone(), ref_gt_bboxes[0].cpu().clone(), self.strides[lvl]
                )
            else:
                raise NotImplementedError
            ref_feats_all.append(_ref_feats)
        return tuple(ref_feats_all)

    def prepare_memory_train_single_level(self, x, gt_bboxes=None, stride=None):
        if self.pixel_sampling_train == 'random':
            _ref_feats = self.get_feats_randomly_single_level(x)
        elif self.pixel_sampling_train == 'bbox':
            _ref_feats = self.get_ref_feats_from_gtbboxes_single_level_train(
                x.clone(), gt_bboxes[0].cpu().clone(), stride
            )
        else:
            raise NotImplementedError

        return _ref_feats

    @staticmethod
    def filter_with_mask(query, mask=None):
        if mask is None:
            return query
        else:
            return query[mask]

    @staticmethod
    def update_with_query(_input, query_new, mask=None):
        if mask is None:
            return query_new
        else:
            _input[mask] = query_new
            return _input

    def forward_train(self, x, ref_x,
                      gt_bboxes=None,
                      ref_gt_bboxes=None):
        assert len(x) == len(self.memories)

        # save ref feats to all levels of memory
        if len(ref_gt_bboxes[0]) < 1:
            print(len(ref_gt_bboxes))
        # ref_feats_all_levels = self.prepare_memory_train(ref_x, ref_gt_bboxes)

        output_all_levels = []
        for lvl, _feat_of_current_level in enumerate(x):
            if lvl < self.start_level:
                # do noting
                output_all_levels.append(_feat_of_current_level)
                continue

            _ref_feats_of_current_level = ref_x[lvl]
            batch_size = _feat_of_current_level.size(0)
            feats_all_imgs_in_batch = []
            for _ind_in_batch in range(batch_size):
                # [1, c, h, w]
                _feat_single_img = _feat_of_current_level[_ind_in_batch: _ind_in_batch+1]
                # [2, c, h, w]
                _ref_feat_single_img = _ref_feats_of_current_level[2*_ind_in_batch: 2*_ind_in_batch+2]
                # do aggregation
                n, c, h, w = _feat_single_img.size()
                # [n, c, h, w] -> [n*h*w, c]
                _feat_single_img = _feat_single_img.permute(0, 2, 3, 1).view(-1, c).contiguous()
                _query = self.filter_with_mask(_feat_single_img)
                _key = self.prepare_memory_train_single_level(_ref_feat_single_img,
                                                              ref_gt_bboxes,
                                                              self.strides[lvl])
                _query_new = self.memories[lvl](_query, _key)
                _output_single_img = self.update_with_query(_feat_single_img, _query_new)
                # [n*h*w, c] -> [n, c, h, w]
                _output_single_img = _output_single_img.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
                feats_all_imgs_in_batch.append(_output_single_img)
            output_all_imgs_in_batch = torch.cat(feats_all_imgs_in_batch, dim=0)
        output_all_levels.append(output_all_imgs_in_batch)

        return tuple(output_all_levels)

    def get_feats_inside_bboxes_single_level(self, x, bboxes, stride):
        """
        save pixels within detected boxes into memory
        :param x: [N, C, H, W]
        :param bboxes: list of ndarray N x [30, 5]
        :param stride: stride of the current level
        :return
        """
        assert len(x) == len(bboxes)

        n, c, _, w = x.size()
        feats_list = []
        for i, _x in enumerate(x):
            ref_obj_list = []

            # [C, H, W] -> [H*W, C]
            _x = _x.view(c, -1).permute(1, 0).contiguous()
            # get bboxes of this image
            _bboxes = bboxes[i]     # 30 x [n, 5]
            for cls_ind in range(len(_bboxes)):
                _bboxes_of_cls = _bboxes[cls_ind]
                # get feats inside high-quality bboxes
                for box_with_score in _bboxes_of_cls:
                    if box_with_score[-1] > 0.3:
                        box = box_with_score[:4]
                        box = (box / stride).astype(int).tolist()
                        inds = sorted(self.box_to_inds_list(box, w))
                        inds = np.asarray(inds)
                        # save part obj
                        PIXEL_NUM = 300
                        if len(inds) > PIXEL_NUM:
                            inds = np.random.choice(inds, PIXEL_NUM, replace=False)
                        inds = np.clip(inds, 0, len(_x)-1)
                        ref_obj_list.append(_x[inds])

            # no high-quality bbox
            IRR_PIXEL_NUM = 50
            if len(ref_obj_list) == 0:
                # obj irr pixels
                obj_irr_inds = self.get_obj_irr_inds_topk(_x, IRR_PIXEL_NUM)
                obj_feats = _x[obj_irr_inds]
            else:
                obj_feats = torch.cat(ref_obj_list, dim=0)

            # objects too small
            if len(obj_feats) == 0:
                obj_irr_inds = self.get_obj_irr_pixels_topk(_x, IRR_PIXEL_NUM)
                obj_feats = _x[obj_irr_inds]

            # max num of feats is set to 1000
            feats_list.append(
                obj_feats[:1000]
            )

        return torch.cat(feats_list, dim=0)

    def write_operation(self, x, bboxes):
        """
        Write features inside bboxes into memory
        :param x: list of feature maps
        :param bboxes: list of bboxes
        :return:
        """
        assert len(x) == len(self.memories)     # number of levels
        assert len(x[0]) == len(bboxes)         # number of frames

        # write for every level
        for lvl, _x in enumerate(x):
            if lvl < self.start_level:
                continue
            # [fix] the memory increasing bug when evaluation
            # [fix] _device = _x.device
            _feats = self.get_feats_inside_bboxes_single_level(
                # [fix] _x.cpu(), bboxes, self.strides[lvl]
                _x, bboxes, self.strides[lvl]
            )
            # update memory
            # [fix] self.memories[lvl].update(_feats.to(_device))
            self.memories[lvl].update(_feats)
        return

    def forward_test(self, x):
        assert len(x) == len(self.memories)

        # do aggregation
        outputs = []
        for lvl, _x in enumerate(x):
            if lvl < self.start_level:
                outputs.append(_x)
                continue
            # [1, C, H, W]
            n, c, h, w = _x.size()
            # [n, c, h, w] -> [n*h*w, c]
            _x = _x.permute(0, 2, 3, 1).view(-1, c).contiguous()

            _query = self.filter_with_mask(_x)
            # query = _x[_mask]
            _key = self.memories[lvl].sample()
            _query_new = self.memories[lvl](_query, _key)
            _output = self.update_with_query(_x, _query_new)
            # _x[_mask] = query_new

            # [n*h*w, c] -> [n, c, h, w]
            _output = _output.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
            outputs.append(_output)

        return tuple(outputs)

    def reset(self):
        for lvl in range(len(self.memories)):
            if lvl < self.start_level:
                continue
            self.memories[lvl].reset()
