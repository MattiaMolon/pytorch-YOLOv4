from torch._C import device
import torch.nn as nn
from tool.torch_utils import *
import torch
from typing import List
import numpy as np


def yolo_BEV_flat_forward(
    prediction: torch.Tensor,
    num_anchors: int,
    num_predictors: int,
) -> torch.Tensor:
    """Forward pass of Yolo_BEV_flat, obtain bboxes inside feature map.

    Args:
        prediction (torch.Tensor): prediction from the last convolutional layer
        num_anchors (int): num of anchors used
        num_predictors (int): num of predictions on each cell

    Returns:
        List[np.ndarray]: List of all bounding boxes, structure define as [batch, W*num_anchors, bbox_attrib].
    """
    # params
    # prediction.shape = [batch_size, filters, 1, W]
    batch_size = prediction.shape[0]
    row_size = prediction.shape[-1]
    bbox_attrib = 7  # dx,dy,dw,dh,sin,cos,obj

    # transform prediction to shape[batch_size, num_predictors(H)*W, bbox_attrib]
    # for each cell I order the bboxes from the bottom of the column to top
    prediction = prediction.view(batch_size, num_predictors * num_anchors * bbox_attrib, row_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, row_size * num_predictors * num_anchors, bbox_attrib)

    # first element: bx = (sigm(tx) + Cx)*cell_angle -> cell_angle applied in post processing
    prediction[..., 0:1] = torch.sigmoid(prediction[..., 0:1])
    x_offset = torch.from_numpy(np.arange(row_size)).float()
    x_offset = (
        x_offset.view(-1, 1)
        .repeat(1, num_predictors * num_anchors)
        .view(num_predictors * num_anchors * row_size, 1)
    )
    prediction[..., 0:1] += x_offset

    # element 2: by = (sigm(ty) + Cy)*cell_depth -> cell_depth applied in post processing
    prediction[..., 1:2] = torch.sigmoid(prediction[..., 1:2])
    y_offset = torch.from_numpy(np.arange(num_predictors)).float()
    y_offset = y_offset.repeat(row_size, num_anchors).view(num_predictors * num_anchors * row_size, 1)
    prediction[..., 1:2] += y_offset

    # elements 3-4: bwl = exp(twl)*pwl  -> pwl applied in post processing
    prediction[..., 2:4] = torch.exp(prediction[..., 2:4])

    # elements 4-5: sin(angle) and cos(angle)
    prediction[..., 4:6] = torch.tanh(prediction[..., 4:5])

    # element 6: obj = sigm(tobj)
    prediction[..., 6:] = torch.sigmoid(prediction[..., 6:])

    return prediction


class YoloBEVFlatLayer(nn.Module):
    """Yolo BEV flat layer
    read the last convolutional layer and return the bboxes in the BEV space
    """

    def __init__(
        self,
        num_classes=0,
        num_anchors=1,
        num_predictors=4,
        model_out=False,
    ):
        super(YoloBEVFlatLayer, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_predictors = num_predictors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output

        # Apply activation functions over predictions
        return yolo_BEV_flat_forward(output, self.num_anchors, self.num_predictors)
