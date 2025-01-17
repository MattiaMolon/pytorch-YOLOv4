from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tool.region_loss import RegionLoss
from tool.yolo_layer import YoloLayer
from tool.yolo_BEV_grid_layer import YoloBEVGridLayer
from tool.yolo_BEV_flat_layer import YoloBEVFlatLayer
from tool.config import *
from tool.torch_utils import *
from typing import List


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        """
        darknet output_size = (input_size + p - k) / s +1
        p : padding = k - 1
        k : size
        s : stride
        torch output_size = (input_size + 2*p -k) / s +1
        p : padding = k//2
        """
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            padding1 = (self.size - 1) // 2
            padding2 = padding1 + 1
        else:
            padding1 = (self.size - 1) // 2
            padding2 = padding1
        if ((x.shape[3] - 1) // self.stride) != ((x.shape[3] + 2 * p - self.size) // self.stride):
            padding3 = (self.size - 1) // 2
            padding4 = padding3 + 1
        else:
            padding3 = (self.size - 1) // 2
            padding4 = padding3
        x = F.max_pool2d(
            F.pad(x, (padding3, padding4, padding1, padding2), mode="replicate"),
            self.size,
            stride=self.stride,
        )
        return x


class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x: torch.tensor):
        assert x.data.dim() == 4

        x = (
            x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1)
            .expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride)
            .contiguous()
            .view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)
        )

        return x


class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4

        out = F.interpolate(x, size=(x.size(2) * self.stride, x.size(3) * self.stride), mode="nearest")
        return out


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, model_type="Yolov4", inference=False):
        """Initialization of Darket based network

        Args:
            cfgfile (str): path to cfg file
            model_type (str, optional): Type of network used. Can be ["Yolov4", "BEV_grid", "BEV_flat"]. Defaults to "Yolov4".
            inference (bool, optional): Define how the network will be used. Defaults to False.
        """
        super(Darknet, self).__init__()

        # what are we using the model for
        self.inference = inference
        self.training = not self.inference

        # define type of model
        self.model_type = model_type

        # read cfg file and divide it into blocks
        self.blocks = parse_cfg(cfgfile)

        # get net params
        self.width = int(self.blocks[0]["width"])
        self.height = int(self.blocks[0]["height"])
        self.cell_depth = int(self.blocks[0]["cell_depth"])
        self.cell_angle = float(self.blocks[0]["cell_angle"])
        # anchors are defined [W,H] for Yolov4 and [H,W] for other models
        self.anchors = [float(i) for i in self.blocks[0]["anchors"].split(",")]
        self.num_anchors = int(self.blocks[0]["num"])
        if self.model_type == "BEV_flat":
            self.num_predictors = int(self.blocks[0]["num_predictors"])

        # create model from blocks
        self.model = self.create_network(self.blocks)
        self.loss = self.model[len(self.model) - 1]

        # needed for weights
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """forward pass of the model

        Args:
            x (np.ndarray): batch of images

        Returns:
            torch.Tensor: bounding boxes predictions
        """

        ind = -2
        concat_output = False
        self.loss = None
        outputs = dict()
        out_boxes = []

        for block in self.blocks:
            ind = ind + 1

            if block["type"] == "net":
                continue

            elif block["type"] in ["convolutional", "maxpool", "upsample"]:
                x = self.model[ind](x)
                outputs[ind] = x

            elif block["type"] == "route":
                layers = block["layers"].split(",")
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    print("rounte number > 2 ,is {}".format(len(layers)))

            elif block["type"] == "shortcut":
                from_layer = int(block["from"])
                activation = block["activation"]
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == "leaky":
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == "relu":
                    x = F.relu(x, inplace=True)
                outputs[ind] = x

            elif block["type"] == "yolo":
                boxes = self.model[ind](x)
                if self.training:  # no preprocessing on output
                    out_boxes.append(boxes)
                else:  # returned preprocessed bboxes
                    if not concat_output:
                        out_boxes = boxes
                        concat_output = True
                    else:
                        out_boxes = torch.cat((out_boxes, boxes), 1)

            else:
                print("unknown type %s" % (block["type"]))

        if self.training:
            # if Yolov4 or BEV_grid
            # out_boxes = [predictions_stride_8,
            #              predictions_stride_16,
            #              predictions_stride_32]
            #
            # if BEV_flat
            # out_boxes = [predictions last layer]
            return out_boxes
        else:
            if self.model_type == "Yolov4":
                return get_region_boxes(out_boxes)

            elif self.model_type in ["BEV_grid", "BEV_flat"]:
                # out_boxes = [x on stride 8,
                #             y on stride 8,
                #             exp(w),
                #             exp(l),
                #             sin(rotation),
                #             cos(rotation),
                #             obj,
                #             ... class confs,
                #             ]
                # transform out_boxes to BEV dimensions such that we can compare them with gt
                return self.transform_pred_to_BEV(out_boxes)

            else:
                print("model type not recognized")
                quit(1)

    def transform_pred_to_BEV(self, out_boxes: torch.Tensor) -> torch.Tensor:
        """Transform out_boxes from biggest feature map space into BEV space

        Args:
            out_boxes (torch.Tensor): concatenated output of yolo_BEV layers

        Output:
            bboxes in BEV space
        """
        # transform anchors dimensions in BEV space
        tmp_anchors = [(self.anchors[i], self.anchors[i + 1]) for i in range(0, len(self.anchors), 2)]
        tmp_anchors = torch.FloatTensor(tmp_anchors)
        tmp_anchors = tmp_anchors.repeat(out_boxes.shape[1] // self.num_anchors, 1).unsqueeze(0)
        out_boxes[..., 2:4] *= tmp_anchors
        del tmp_anchors

        # transform anchors distance and angle into BEV space
        out_boxes[..., 0] *= self.cell_angle  # row
        out_boxes[..., 1] *= self.cell_depth  # column

        # adjust alpha to be consistent with dataset
        fovx = (len(out_boxes[0, :]) // self.num_predictors) * self.cell_angle
        out_boxes[..., 0] -= fovx / 2

        return out_boxes

    def print_network(self):
        print_cfg(self.blocks)

    def freeze_layers(self, layer_ids: List[int]):
        """Freeze layers with specified IDs during training

        Args:
            layer_ids (List[int]): layer IDs to freeze
        """
        for id in layer_ids:
            for param in self.model[id].parameters():
                param.requires_grad = False

    def create_network(self, blocks):
        model = nn.ModuleList()

        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:

            if block["type"] == "net":
                prev_filters = int(block["channels"])
                continue

            elif block["type"] == "convolutional":
                conv_id += 1
                batch_normalize = int(block["batch_normalize"])
                filters = int(block["filters"])
                is_pad = int(block["pad"])
                stride = int(block["stride"])
                kernel_size = [int(x) for x in block["size"].strip().split(",")]
                if len(kernel_size) == 2:  # When this happens kernel size must be odd in both dimensions
                    kernel_size = (kernel_size[0], kernel_size[1])  # pytorch convention H x W
                    pad = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) if is_pad else 0
                else:
                    kernel_size = (kernel_size[0], kernel_size[0])
                    pad = (
                        (kernel_size[0] - 1) // 2 if is_pad else 0
                    )  # padding is defined as size/2 in yolo wiki
                activation = block["activation"]

                # add convolutional layer
                conv = nn.Sequential()
                if batch_normalize:
                    conv.add_module(
                        "conv{0}".format(conv_id),
                        nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False),
                    )
                    conv.add_module("bn{0}".format(conv_id), nn.BatchNorm2d(filters))
                else:
                    conv.add_module(
                        "conv{0}".format(conv_id),
                        nn.Conv2d(prev_filters, filters, kernel_size, stride, pad),
                    )

                # add activation function
                if activation == "leaky":
                    conv.add_module("leaky{0}".format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == "relu":
                    conv.add_module("relu{0}".format(conv_id), nn.ReLU(inplace=True))
                elif activation == "mish":
                    conv.add_module("mish{0}".format(conv_id), Mish())
                elif activation == "linear":
                    pass
                else:
                    print(f'conv{conv_id} activation "{activation}" not recognized')

                # update params
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                model.append(conv)

            elif block["type"] == "maxpool":
                pool_size = int(block["size"])
                stride = int(block["stride"])

                if stride == 1 and pool_size % 2:
                    maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    maxpool = MaxPoolDark(pool_size, stride)

                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                model.append(maxpool)

            elif block["type"] == "upsample":
                stride = int(block["stride"])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)

                model.append(Upsample_expand(stride))
                # model.append(Upsample_interpolate(stride))

            elif block["type"] == "route":
                layers = block["layers"].split(",")
                ind = len(model)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert layers[0] == ind - 1 or layers[1] == ind - 1
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    assert layers[0] == ind - 1
                    prev_filters = (
                        out_filters[layers[0]]
                        + out_filters[layers[1]]
                        + out_filters[layers[2]]
                        + out_filters[layers[3]]
                    )
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                model.append(EmptyModule())

            elif block["type"] == "shortcut":
                ind = len(model)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                model.append(EmptyModule())

            elif block["type"] == "yolo":
                if self.model_type == "BEV_grid":  # BEV grid layer
                    # TODO: go through this code and remove unneccessary parameters
                    yolo_layer = YoloBEVGridLayer()
                    yolo_layer.num_classes = int(block["classes"])
                    self.num_classes = yolo_layer.num_classes
                    yolo_layer.num_anchors = self.num_anchors
                    yolo_layer.stride = prev_stride
                    yolo_layer.scale_x_y = float(block["scale_x_y"])
                    out_filters.append(prev_filters)
                    out_strides.append(prev_stride)
                    model.append(yolo_layer)

                elif self.model_type == "BEV_flat":  # BEV flat layer
                    yolo_layer = YoloBEVFlatLayer()
                    yolo_layer.num_classes = int(block["classes"])
                    self.num_classes = yolo_layer.num_classes
                    yolo_layer.num_predictors = self.num_predictors
                    yolo_layer.num_anchors = self.num_anchors
                    model.append(yolo_layer)

                elif self.model_type == "Yolov4":  # Classic Yolo layer
                    yolo_layer = YoloLayer()
                    anchors = block["anchors"].split(",")
                    anchor_mask = block["mask"].split(",")
                    yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                    yolo_layer.anchors = [float(i) for i in anchors]
                    yolo_layer.num_classes = int(block["classes"])
                    self.num_classes = yolo_layer.num_classes
                    yolo_layer.num_anchors = int(block["num"])
                    yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                    yolo_layer.stride = prev_stride
                    yolo_layer.scale_x_y = float(block["scale_x_y"])
                    out_filters.append(prev_filters)
                    out_strides.append(prev_stride)
                    model.append(yolo_layer)

                else:
                    print("model type not recognized while building yolo layer!")

            else:
                print("unknown type %s" % (block["type"]))

        return model

    def load_weights(self, weightfile: str, skip_layers: List[int] = [], cut_off: int = 10000):
        """load weights from darknet file format
        To have a better understanding of .weights definition look at:
        https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
        For parameters that require layers indexes -> You can look at the layers' indexes using the print_network() function.

        Args:
            weightfile (str): Path to weights file
            skip_layers (List[int]): list of layers' indexes to skip while loading weights. Default = []
            cut_off (int): index of the last layer's weights to load. Default = 10000.
        """
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            # end condition
            if start >= buf.size:
                break

            # skip conditions
            ind = ind + 1
            if ind > cut_off:
                break
            if ind in skip_layers:
                print(f"WARNING: Skipped layer {ind} while loading weights")
                continue

            # load weights
            if block["type"] == "net":
                continue
            elif block["type"] == "convolutional":
                model = self.model[ind]
                batch_normalize = int(block["batch_normalize"])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block["type"] == "maxpool":
                pass
            elif block["type"] == "upsample":
                pass
            elif block["type"] == "route":
                pass
            elif block["type"] == "shortcut":
                pass
            elif block["type"] == "yolo":
                pass
            else:
                print("unknown type %s" % (block["type"]))

    def save_weights(self, outfile, cutoff=0):
        """Save weights in a Darknet fashion

        Args:
            outfile (str): Name of the .weights file
            cutoff (int, optional): Number of blocks we don't want to include in the saved weights,
            starting from the last layer. Defaults to 0.
        """
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1

        fp = open(outfile, "wb")
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff + 1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block["type"] == "convolutional":
                model = self.model[ind]
                batch_normalize = int(block["batch_normalize"])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block["type"] == "maxpool":
                pass
            elif block["type"] == "upsample":
                pass
            elif block["type"] == "route":
                pass
            elif block["type"] == "shortcut":
                pass
            elif block["type"] == "yolo":
                pass
            else:
                print("unknown type %s" % (block["type"]))
        fp.close()
