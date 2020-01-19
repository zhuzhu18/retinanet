import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]     # 从P3到P7层anchors的大小
        if ratios is None:
            self.ratios = np.array([1.0/2.0, 1.0/1.0, 2.0/1.0])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):

        image_shape = image.shape[2:]     # torch.Size([H, W])
        image_shape = np.array(image_shape)
        feature_shapes = [(image_shape + stride - 1) // stride for stride in self.strides]    # 每一层的高和宽是8, 16, 32, 64, 128的整数倍

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(feature_shapes[idx], self.strides[idx], anchors)       # (K*A, 4)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)       # (5*K*A, 4)   # 5个level的feature所有位置anchors坐标

        all_anchors = np.expand_dims(all_anchors, axis=0)       # (1, 5*K*A, 4)

        return torch.from_numpy(all_anchors).cuda()


def generate_anchors(base_size=32, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]      # 9个anchors的面积

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        ratios=None,
        scales=None,
        strides=None,
        sizes=None,
        shapes_callback=None,
):
    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    H, W = shape
    A = anchors.shape[0]
    shift_x, shift_y = np.meshgrid(stride * (np.arange(W)+0.5), stride * (np.arange(H)+0.5))

    anchors_x1_x2 = anchors[:, 0::2].reshape((1, 1, A, -1)) + shift_x.reshape((H, W, 1, 1))
    anchors_y1_y2 = anchors[:, 1::2].reshape((1, 1, A, -1)) + shift_y.reshape((H, W, 1, 1))

    all_anchors = np.vstack((anchors_x1_x2[:, :, :, 0].ravel(), anchors_y1_y2[:, :, :, 0].ravel(),
                             anchors_x1_x2[:, :, :, 1].ravel(), anchors_y1_y2[:, :, :, 1].ravel())).T
    return all_anchors

