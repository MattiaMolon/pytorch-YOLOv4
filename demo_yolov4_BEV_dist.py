from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from dataset import Yolo_BEV_dataset, Yolo_dataset
import cv2
import argparse

"""hyper parameters"""
# check for cuda
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def detect_BEV_dist(cfgfile, weightfile, imgfile):
    """Detect elements in BEV map with yolov4_BEV_flat

    Args:
        cfgfile (str): Path to .cfg file
        weightfile (str): Path to .weights file
        imgfile (str): Path to image on which we want to run BEV detection
    """

    # load model
    m = Darknet(cfgfile, model_type="BEV_dist")
    m.print_network()

    if weightfile.endswith(".weights"):
        m.load_weights(weightfile, cut_off=85)
    else:
        m.load_state_dict(torch.load(weightfile, map_location=torch.device(device)))
    print("Loading backbone from %s... Done!" % (weightfile))

    # push to GPU
    m.to(device)

    # read sample image
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
    img = cv2.resize(img, (int(m.blocks[0]["width"]), int(m.blocks[0]["height"])))
    img /= 255.0

    # create batch
    batch = np.expand_dims(img, 0)

    # run inference
    start = time.time()
    boxes = do_detect(m, batch, device)
    finish = time.time()
    print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))


def get_args():
    parser = argparse.ArgumentParser("Test your image by trained model.")
    parser.add_argument(
        "-cfg",
        "--cfgfile",
        type=str,
        default="./cfg/model/yolov4_BEV_dist_nuScenes.cfg",
        help="path of cfg file",
        dest="cfgfile",
    )
    parser.add_argument(
        "-l",
        "--load",
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
    detect_BEV_dist(args.cfgfile, args.weightfile, args.imgfile)
