import torch.nn as nn
from tool.torch_utils import *
import torch
from typing import List
import numpy as np


def yolo_BEV_dist_forward(
    prediction: torch.Tensor,
    num_predictors: int,
    device: str,
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

    # transform prediction to shape[batch_size, W, num_predictors]
    # for each cell I order the bboxes from the bottom of the column to top
    prediction = prediction.view(batch_size, num_predictors, row_size)
    prediction = prediction.transpose(1, 2).contiguous()

    # apply sigmoid
    prediction = torch.sigmoid(prediction)

    return prediction


class YoloBEVDistLayer(nn.Module):
    """Yolo BEV dist layer
    read the last convolutional layer and return the bboxes in the BEV space
    """

    def __init__(self, num_classes=0, num_predictors=4, model_out=False, device="cpu"):
        super(YoloBEVDistLayer, self).__init__()
        self.num_classes = num_classes
        self.num_predictors = num_predictors
        self.device = device

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output

        # Apply activation functions over predictions
        return yolo_BEV_dist_forward(output, self.num_predictors, self.device)
