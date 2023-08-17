# Copyright (c) OpenMMLab. All rights reserved.
from .auto_augment import (AutoAugment, BrightnessTransform, ColorTransform,
                           ContrastTransform, EqualizeTransform, Rotate, Shear,
                           Translate)
from .compose import Compose
from .formating import (Collect, DefaultFormatBundle, ImageToTensor,
                        ToDataContainer, ToTensor, Transpose, to_tensor)
from .instaboost import InstaBoost
from .loading import (LoadAnnotations, LoadImageFromFile, LoadImageFromWebcam,
                      LoadMultiChannelImageFromFiles, LoadProposals)
from .test_time_aug import MultiScaleFlipAug
from .transforms import (Albu, CutOut, Expand, MinIoURandomCrop, MixUp, Mosaic,
                         Normalize, Pad, PhotoMetricDistortion, RandomAffine,
                         RandomCenterCropPad, RandomCrop, RandomFlip,
                         RandomShift, Resize, SegRescale, YOLOXHSVRandomAug)
# add from mmtrack
from .mmtrack.formatting import (
    ConcatVideoReferences,
    ReIDFormatBundle,
    SeqDefaultFormatBundle,
    ToList,
    VideoCollect,
)
from .mmtrack.loading import LoadDetections, LoadMultiImagesFromFile, SeqLoadAnnotations
from .mmtrack.processing import MatchInstances
from .mmtrack.transforms import (
    SeqBlurAug,
    SeqColorAug,
    SeqCropLikeSiamFC,
    SeqNormalize,
    SeqPad,
    SeqPhotoMetricDistortion,
    SeqRandomCrop,
    SeqRandomFlip,
    SeqResize,
    SeqShiftScaleAug,
    SeqRandomCenterCropPad,
)

__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer',
    'Transpose', 'Collect', 'DefaultFormatBundle', 'LoadAnnotations',
    'LoadImageFromFile', 'LoadImageFromWebcam',
    'LoadMultiChannelImageFromFiles', 'LoadProposals', 'MultiScaleFlipAug',
    'Resize', 'RandomFlip', 'Pad', 'RandomCrop', 'Normalize', 'SegRescale',
    'MinIoURandomCrop', 'Expand', 'PhotoMetricDistortion', 'Albu',
    'InstaBoost', 'RandomCenterCropPad', 'AutoAugment', 'CutOut', 'Shear',
    'Rotate', 'ColorTransform', 'EqualizeTransform', 'BrightnessTransform',
    'ContrastTransform', 'Translate', 'RandomShift', 'Mosaic', 'MixUp',
    'RandomAffine', 'YOLOXHSVRandomAug',
    # mmtrack pipelines
    "LoadMultiImagesFromFile",
    "SeqLoadAnnotations",
    "SeqResize",
    "SeqNormalize",
    "SeqRandomFlip",
    "SeqPad",
    "SeqDefaultFormatBundle",
    "VideoCollect",
    "ConcatVideoReferences",
    "LoadDetections",
    "MatchInstances",
    "SeqRandomCrop",
    "SeqPhotoMetricDistortion",
    "SeqCropLikeSiamFC",
    "SeqShiftScaleAug",
    "SeqBlurAug",
    "SeqColorAug",
    "ToList",
    "ReIDFormatBundle",
    "SeqRandomCenterCropPad",
]