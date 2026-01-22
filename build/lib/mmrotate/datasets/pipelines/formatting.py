# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.pipelines.formatting import DefaultFormatBundle

from ..builder import ROTATED_PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@ROTATED_PIPELINES.register_module()
class SarDefaultFormatBundle(DefaultFormatBundle):

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        super(SarDefaultFormatBundle, self).__init__()
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                img = img.astype(np.float32)
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            results['img'] = DC(
                img, padding_value=self.pad_val['img'], stack=True)

        if 'img_stft' in results:
            img_stft = results['img_stft']

            if not img_stft.flags.c_contiguous:
                img_stft = np.ascontiguousarray(img_stft)
                img_stft = to_tensor(img_stft)
            else:
                img_stft = to_tensor(img_stft).contiguous()
            results['img_stft'] = DC(
                img_stft, padding_value=self.pad_val['img'], stack=True)

        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results
