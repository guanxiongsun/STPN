# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                      ROI_EXTRACTORS, SHARED_HEADS, build_backbone,
                      build_detector, build_head, build_loss, build_neck,
                      build_roi_extractor, build_shared_head,
                      # from mmtrack
                      build_model, build_aggregator, AGGREGATORS,
                      # for memory
                      build_memory,
                      # for predictor
                      PREDICTORS, build_predictor,
                      )
from .dense_heads import *  # noqa: F401,F403
from .detectors import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .plugins import *  # noqa: F401,F403
from .roi_heads import *  # noqa: F401,F403
from .seg_heads import *  # noqa: F401,F403

# add video models
from .vid import SELSA, FCOSAtt, CenterNetAtt, SELSAVideoPrompt
from .aggregators import SelsaAggregator
from .memory import MPN

__all__ = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'DETECTORS', 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector',
    # add wrapper for video models
    'build_model', 'SELSA', 'AGGREGATORS', 'build_aggregator',
    # build memory
    'build_memory', 'FCOSAtt', 'MPN', 'CenterNetAtt',
    'PREDICTORS', 'build_predictor',
]
