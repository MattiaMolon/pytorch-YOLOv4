import torch.nn as nn
from tool.torch_utils import *
import torch
from typing import List
import numpy as np


def yolo_BEV_forward(
    prediction: torch.Tensor,
    stride: int,
    num_anchors: int,
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
        List[np.ndarray]: List of all bounding boxes, structure define as [batch, H*W*num_anchors, bbox_attrib].
            The bboxes returned are scaled on the biggest feature map in the network. Default stride = 8.
    """

    # params
    batch_size = prediction.shape[0]
    grid_size = prediction.shape[2:]  # [H, W]
    bbox_attrib = 7 + num_classes  # dx,dy,dw,dl,sin,cos,obj + classes

    # transform prediction to shape[batch_size, num_anchors*H*W, bbox_attrib]
    prediction = prediction.view(
        batch_size, bbox_attrib * num_anchors, grid_size[1] * grid_size[0]
    )
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size[1] * grid_size[0] * num_anchors, bbox_attrib
    )

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

    # elements 3-4: bwl = exp(twl)*pwl  -> pwl is applied in the forward pass
    prediction[..., 2:4] = torch.exp(prediction[..., 2:4])

    # elements 4-5: sin/cos = tanh(tsin/tcos)
    prediction[..., 4:6] = torch.tanh(prediction[..., 4:5])

    # element 6: obj = sigm(tobj)
    prediction[..., 6] = torch.sigmoid(prediction[..., 6])

    # all remaining: class_prob = sigm(class)
    prediction[..., 7:] = torch.sigmoid(prediction[..., 7:])

    # reshape coordinates to smallest stride. Smallest: 8, strides: 8, 16, 32
    prediction[..., :2] *= stride / smallest_stride

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
        self.num_classes = num_classes
        self.num_anchors = 1
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
            print("training not implemented yet")
            exit(0)
            return output

        return yolo_BEV_forward(output, self.stride, self.num_anchors, self.num_classes)
