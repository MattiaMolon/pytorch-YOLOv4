# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from torchsummary import summary
import torch
import argparse

"""hyper parameters"""
use_cuda = False


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2

    m = Darknet(cfgfile)
    # m.print_network()
    m.load_weights(weightfile)
    print("Loading weights from %s... Done!" % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = "data/voc.names"
    elif num_classes == 80:
        namesfile = "data/coco.names"
    else:
        namesfile = "data/x.names"
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename="predictions.jpg", class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser("Test your image by trained model.")
    parser.add_argument(
        "-cfgfile",
        type=str,
        default="./cfg/yolov4.cfg",
        help="path of cfg file",
        dest="cfgfile",
    )
    parser.add_argument(
        "-weightfile",
        type=str,
        default="./checkpoints/yolov4.weights",
        help="path of trained model.",
        dest="weightfile",
    )
    parser.add_argument(
        "-imgfile",
        type=str,
        default="../data/KITTI/training/images/000001.png",
        help="path of your image file.",
        dest="imgfile",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
