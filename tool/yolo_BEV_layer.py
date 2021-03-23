import torch.nn as nn
import torch.nn.functional as F
from traitlets.traitlets import Bool
from tool.torch_utils import *
import torch
from typing import List
import numpy as np


def yolo_forward_dynamic(
    output,
    conf_thresh,
    num_classes,
    anchors,
    num_anchors,
    scale_x_y,
    only_objectness=1,
    validation=False,
):
    # Output would be invalid if it does not satisfy this assert
    # assert output.size(1) == (5 + num_classes) * num_anchors

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size(0)
    # H = output.size(2)
    # W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)

        bxy_list.append(output[:, begin : begin + 2, ...])
        bwh_list.append(output[:, begin + 2 : begin + 4, ...])
        det_confs_list.append(output[:, begin + 4 : begin + 5, ...])
        cls_confs_list.append(output[:, begin + 5 : end, ...])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(
        output.size(0), num_anchors * output.size(2) * output.size(3)
    )

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(
        output.size(0), num_anchors, num_classes, output.size(2) * output.size(3)
    )
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes]
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(
        output.size(0), num_anchors * output.size(2) * output.size(3), num_classes
    )

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(
        np.expand_dims(
            np.expand_dims(
                np.linspace(0, output.size(3) - 1, output.size(3)), axis=0
            ).repeat(output.size(2), 0),
            axis=0,
        ),
        axis=0,
    )
    grid_y = np.expand_dims(
        np.expand_dims(
            np.expand_dims(
                np.linspace(0, output.size(2) - 1, output.size(2)), axis=1
            ).repeat(output.size(3), 1),
            axis=0,
        ),
        axis=0,
    )
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii : ii + 1] + torch.tensor(
            grid_x, device=device, dtype=torch.float32
        )  # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(
            grid_y, device=device, dtype=torch.float32
        )  # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)

    ########################################
    #   Figure out bboxes from slices     #
    ########################################

    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    by = by_bh[:, :num_anchors].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    bw = bx_bw[:, num_anchors:].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    bh = by_bh[:, num_anchors:].view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4
    )
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(
        output.size(0), num_anchors * output.size(2) * output.size(3), 1
    )
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return boxes, confs


def yolo_BEV_forward(
    prediction: torch.Tensor,
    stride: int,
    masked_anchors: List[int],
    num_classes: int,
    smallest_stride: int = 8,
) -> torch.Tensor:
    """forward pass of Yolo_BEV, return all predicted bounding boxes in the BEV space

    Args:
        prediction (torch.Tensor): prediction from the last convolutional layer
        stride (int): stride of the feature map
        masked_anchors (List[float]): anchors for this layer
        num_classes (int): num of classes to predict
        smallest_stride (int): smallest stride in the network. Default = 8.

    Returns:
        List[np.ndarray]: List of all bounding boxes, structure define as [batch, H*W*num_anchors].
            The bboxes returned are scaled on the biggest feature map in the network. Default stride 8 in respect to original image.
    """

    # params
    batch_size = prediction.shape[0]
    grid_size = prediction.shape[2:]  # [H, W]
    bbox_attrib = 7 + num_classes  # dx,dy,dw,dl,sin,cos,obj + classes
    num_anchors = len(masked_anchors)

    # transform prediction to shape[batch_size, num_anchors*H*W]
    prediction = prediction.view(
        batch_size, bbox_attrib * num_anchors, grid_size[1] * grid_size[0]
    )
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size[1] * grid_size[0] * num_anchors, bbox_attrib
    )

    # reduce anchors dimensions to feature maps
    masked_anchors = [(ma[0] / stride, ma[1] / stride) for ma in masked_anchors]

    # first 2 elements: bxy = sigm(txy) + Cxy
    prediction[..., :2] = torch.sigmoid(prediction[..., :2])
    grid_h = np.arange(grid_size[0])
    grid_w = np.arange(grid_size[1])
    a, b = np.meshgrid(grid_w, grid_h)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    x_y_offset = (
        torch.cat((x_offset, y_offset), 1)
        .repeat(1, num_anchors)
        .view(-1, 2)
        .unsqueeze(0)
    )
    prediction[..., :2] += x_y_offset

    # elements 3-4: bwl = exp(twl)*pwl
    masked_anchors = torch.FloatTensor(masked_anchors)
    masked_anchors = masked_anchors.repeat(grid_size[0] * grid_size[1], 1).unsqueeze(0)
    prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * masked_anchors

    # elements 4-5: sin/cos = tanh(tsin/tcos)
    prediction[..., 4:6] = torch.tanh(prediction[..., 4:5])

    # element 6: obj = sigm(tobj)
    prediction[..., 6] = torch.sigmoid(prediction[..., 6])

    # all remaining: class_prob = sigm(class)
    prediction[..., 7:] = torch.sigmoid(prediction[..., 7:])

    # reshape detected bounding boxes to smallest stride. Smallest: 8, strides: 8, 16, 32
    prediction[..., :4] *= stride / smallest_stride

    return prediction


class YoloBEVLayer(nn.Module):
    """Yolo BEV layer
    read the last convolutional layer and return the bboxes in the BEV space
    """

    def __init__(
        self,
        anchor_mask=[],
        num_classes=0,
        anchors=[],
        num_anchors=1,
        stride=32,
        model_out=False,
    ):
        super(YoloBEVLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output

        # if inference time
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[
                m * self.anchor_step : (m + 1) * self.anchor_step
            ]
        masked_anchors = [
            (masked_anchors[i], masked_anchors[i + 1])
            for i in range(0, len(masked_anchors), 2)
        ]

        return yolo_BEV_forward(output, self.stride, masked_anchors, self.num_classes)
