import numpy as np
import torch
import torch.nn as nn


def calc_iou(box1, box2):
    iou = torch.zeros((len(box1), len(box2)))
    for i in range(len(box1)):
        iw = torch.min(box1[i, 2], box2[:, 2]) - torch.max(box1[i, 0], box2[:, 0])
        ih = torch.min(box1[i, 3], box2[:, 3]) - torch.max(box1[i, 1], box2[:, 1])

        inter = torch.clamp(iw, 0) * torch.clamp(ih, 0)
        total_area = (box1[i, 2] - box1[i, 0]) * (box1[i, 3] - box1[i, 1]) + \
                     (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        iou[i] = inter.div(total_area - inter)

    return iou


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]     # 形状为[5*K*A, 4]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]     # (5*H*W*A)*K
            regression = regressions[j, :, :]     # (5*H*W*A)*4

            bbox_annotation = annotations[j, :, :]     # num_annots * 5
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]     # 取出正常标注的样本的标注, valid_num_annots * 5

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)     # (5*H*W*A)*K

            IoU = calc_iou(anchor, bbox_annotation[:, :4])  # num_anchors x valid_num_annots

            # IoU_max表示每个anchor与标注框重叠度最高的那个标注框之间的IoU
            # IoU_argmax表示每个anchor与标注框重叠度最高的那个标注框的索引
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # (num_anchors, )

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1     # (5*H*W*A)*K
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0       # IOU<0.4的设为0

            positive_indices = torch.ge(IoU_max, 0.5)    # IOU>=0.5的anchors的索引的掩码, (num_anchors, )

            num_positive_anchors = positive_indices.sum()      # 正样本的数量

            assigned_annotations = bbox_annotation[IoU_argmax, :]     # (num_anchors, 5), anchors对应的标注框

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1    # 正样本的one-hot向量, (num_anchors, K)

            alpha_factor = torch.full(targets.shape, fill_value=alpha).cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]       # 正样本的anchors的宽度
                anchor_heights_pi = anchor_heights[positive_indices]     # 正样本的anchors的高度
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]         # 正样本的anchors的中心x坐标
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]         # 正样本的anchors的中心y坐标

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi      # (num_anchors, )
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])    # (num_positives, 4)

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0,
                                                                                                                 keepdim=True)


