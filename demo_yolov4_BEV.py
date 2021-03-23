# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import cv2
import argparse

"""hyper parameters"""
use_cuda = False


def detect_BEV(cfgfile, weightfile, imgfile):
    """Detect elements in BEV map with yolov4_BEV

    Args:
        cfgfile (str): Path to .cfg file
        weightfile (str): Path to .weights file
        imgfile (str): Path to image on which we want to run BEV detection
    """

    # load model
    m = Darknet(cfgfile, BEV=True)
    # m.print_network()
    # m.load_weights(weightfile)
    # print("Loading weights from %s... Done!" % (weightfile))

    # push to GPU
    if use_cuda:
        m.cuda()

    # load names
    namesfile = "names/BEV.names"
    class_names = load_class_names(namesfile)

    # read sample image
    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    # run inference
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    # TODO
    # plot_boxes_cv2(img, boxes[0], savename="predictions.jpg", class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser("Test your image by trained model.")
    parser.add_argument(
        "-cfgfile",
        type=str,
        default="./cfg/yolov4_BEV.cfg",
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
    detect_BEV(args.cfgfile, args.weightfile, args.imgfile)
