import sys
import os

sys.path.insert(0, os.path.abspath("."))

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
# check for cuda
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2

    m = Darknet(cfgfile, model_type="Yolov4")
    # m.print_network()
    m.load_weights(weightfile)
    print("Loading weights from %s... Done!" % (weightfile))
    m.to(device)

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = "names/voc.names"
    elif num_classes == 80:
        namesfile = "names/coco.names"
    else:
        namesfile = "names/x.names"
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    input = cv2.resize(img, (m.width, m.height))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    for i in range(1):
        start = time.time()
        boxes = do_detect(m, input, device, 0.4, 0.6)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename="predictions.jpg", class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser("Test your image by trained model.")
    parser.add_argument(
        "-cfgfile",
        type=str,
        default="./cfg/model/yolov4.cfg",
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
        default="/home/mattia/university/Master_thesis/data/nuScenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604012404.jpg",
        help="path of your image file.",
        dest="imgfile",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
