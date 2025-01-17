import torch
from tool.torch_utils import convert2cpu


def parse_cfg(cfgfile):
    """parse .cfg file

    Args:
        cfgfile (str): Path to .cfg file

    Returns:
        list: list of dictionaries {block: info}
    """
    blocks = []
    with open(cfgfile, "r") as fp:
        block = None
        line = fp.readline()
        while line != "":
            line = line.rstrip()
            if line == "" or line[0] == "#":
                line = fp.readline()
                continue
            elif line[0] == "[":
                if block:
                    blocks.append(block)
                block = dict()
                block["type"] = line.lstrip("[").rstrip("]")
                # set default value
                if block["type"] == "convolutional":
                    block["batch_normalize"] = 0
            else:
                key, value = line.split("=")
                key = key.strip()
                if key == "type":
                    key = "_type"
                value = value.strip()
                block[key] = value
            line = fp.readline()

        if block:
            blocks.append(block)

    return blocks


def print_cfg(blocks):
    """Print on screen the structure of the network described in the cfg file

    Args:
        blocks (list): List of blocks extracted by the .cfg file describing every layer of the network
    """
    print("layer     filters    size              input                output")
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2

    for block in blocks:
        ind = ind + 1

        if block["type"] == "net":
            prev_width = int(block["width"])
            prev_height = int(block["height"])
            prev_filters = int(block["channels"])
            continue

        elif block["type"] == "convolutional":
            filters = int(block["filters"])
            kernel_size = [int(x) for x in block["size"].strip().split(",")]
            stride = int(block["stride"])
            is_pad = int(block["pad"])
            if len(kernel_size) == 2:
                kernel_size = (kernel_size[0], kernel_size[1])
                pad = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) if is_pad else (0, 0)
            else:
                kernel_size = (kernel_size[0], kernel_size[0])
                # padding is defined as size-1/2 in yolo wiki
                pad = ((kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2) if is_pad else (0, 0)
            width = (prev_width + 2 * pad[1] - kernel_size[1]) // stride + 1
            height = (prev_height + 2 * pad[0] - kernel_size[0]) // stride + 1
            print(
                "%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    ind,
                    "conv",
                    filters,
                    kernel_size[0],
                    kernel_size[1],
                    stride,
                    prev_height,
                    prev_width,
                    prev_filters,
                    height,
                    width,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block["type"] == "maxpool":
            pool_size = int(block["size"])
            stride = int(block["stride"])
            width = prev_width // stride
            height = prev_height // stride
            print(
                "%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    ind,
                    "max",
                    pool_size,
                    pool_size,
                    stride,
                    prev_height,
                    prev_width,
                    prev_filters,
                    height,
                    width,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block["type"] == "upsample":
            stride = int(block["stride"])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print(
                "%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    ind,
                    "upsample",
                    stride,
                    prev_height,
                    prev_width,
                    prev_filters,
                    height,
                    width,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block["type"] == "route":
            layers = block["layers"].split(",")
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print("%5d %-6s %d" % (ind, "route", layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print("%5d %-6s %d %d" % (ind, "route", layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert prev_width == out_widths[layers[1]]
                assert prev_height == out_heights[layers[1]]
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            elif len(layers) == 4:
                print("%5d %-6s %d %d %d %d" % (ind, "route", layers[0], layers[1], layers[2], layers[3]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert prev_width == out_widths[layers[1]] == out_widths[layers[2]] == out_widths[layers[3]]
                assert (
                    prev_height == out_heights[layers[1]] == out_heights[layers[2]] == out_heights[layers[3]]
                )
                prev_filters = (
                    out_filters[layers[0]]
                    + out_filters[layers[1]]
                    + out_filters[layers[2]]
                    + out_filters[layers[3]]
                )
            else:
                print(
                    "route error !!! {} {} {}".format(
                        sys._getframe().f_code.co_filename,
                        sys._getframe().f_code.co_name,
                        sys._getframe().f_lineno,
                    )
                )

            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block["type"] in ["yolo"]:
            print("%5d %-6s" % (ind, "detection"))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        elif block["type"] == "shortcut":
            from_id = int(block["from"])
            from_id = from_id if from_id > 0 else from_id + ind
            print("%5d %-6s %d" % (ind, "shortcut", from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)

        else:
            print("unknown type %s" % (block["type"]))


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(
        torch.from_numpy(buf[start : start + num_w]).reshape(conv_model.weight.data.shape)
    )
    start = start + num_w
    return start


def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(
        torch.from_numpy(buf[start : start + num_w]).reshape(conv_model.weight.data.shape)
    )
    start = start + num_w
    return start


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start : start + num_b]))
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start : start + num_w]))
    start = start + num_w
    return start


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


if __name__ == "__main__":
    import sys

    blocks = parse_cfg("cfg/yolo.cfg")
    if len(sys.argv) == 2:
        blocks = parse_cfg(sys.argv[1])
    print_cfg(blocks)
