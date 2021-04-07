from numpy.lib.financial import ipmt
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import cv2
import argparse

"""hyper parameters"""
use_cuda = False


def detect_BEV_flat(cfgfile, weightfile, imgfile):
    """Detect elements in BEV map with yolov4_BEV_flat

    Args:
        cfgfile (str): Path to .cfg file
        weightfile (str): Path to .weights file
        imgfile (str): Path to image on which we want to run BEV detection
    """

    # load model
    m = Darknet(cfgfile, model_type="BEV_flat")
    m.print_network()

    m.load_weights(weightfile, cut_off=54)
    print("Loading backbone from %s... Done!" % (weightfile))

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

    # create batch
    sized = np.expand_dims(sized, 0)
    sized = np.concatenate((sized, sized), 0)

    # run inference
    start = time.time()
    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    finish = time.time()
    print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    # TODO: plot boxes in BEV


def get_args():
    parser = argparse.ArgumentParser("Test your image by trained model.")
    parser.add_argument(
        "-cfgfile",
        type=str,
        default="./cfg/model/yolov4_BEV_flat.cfg",
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
        default="../data/KITTI/train/images/000001.png",
        help="path of your image file.",
        dest="imgfile",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    detect_BEV_flat(args.cfgfile, args.weightfile, args.imgfile)
