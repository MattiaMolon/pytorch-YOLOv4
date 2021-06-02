import sys, os

sys.path.insert(0, os.path.abspath("."))

import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import Yolo_BEV_dataset
from cfg.train.cfg_yolov4 import Cfg
from tool.darknet2pytorch import Darknet
from tool.utils import post_processing


def collate(batch):
    images = []
    labels = []
    paths = []
    for img, lab, path in batch:
        images.append([img])
        paths.append(path)
        if len(lab.shape) == 1:
            labels.append(np.expand_dims(lab, 0))
        else:
            labels.append(lab)

    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    return images, labels, paths


def predict_and_save(model, device, config):
    # Get dataloaders
    dataset = Yolo_BEV_dataset(config, split=config.split, input_type="nuScenes")

    data_loader = DataLoader(
        dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )

    # log
    n_samples = len(dataset)
    global_step = 0
    print(
        f"""Starting prediction with Yolov4:
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Training size:   {n_samples}
        Device:          {device.type}
        Input height:    {config.height}
        Input width:     {config.width}
        Dataset classes: {config.classes}
        Grid dimension:  {config.num_columns}x{config.num_predictors}
    """
    )

    # start evaluating
    model.eval()
    im_paths, xmins, ymins, xmaxs, ymaxs, dists, confs = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )
    with tqdm(total=n_samples, desc="Running Yolov4", unit="img", ncols=100) as pbar:

        for batch in data_loader:
            # get batch
            global_step += 1
            images = batch[0].float().to(device)
            paths = batch[2]

            # gather predictions
            # predictions in yolo:
            # bboxes -> [anchors*H*W, (xmin, ymin, xmax, ymax)]
            # confds -> [anchors*H*W, conf_score*class]
            preds = model(images)

            ####### save preds
            # save preds in a csv file
            bboxes_batch = post_processing(config.obj_thresh, config.nms_thresh, preds, verbose=False)
            for bboxes, path in zip(bboxes_batch, paths):
                if len(bboxes) == 0:
                    continue

                # filter bboxes
                bboxes = np.array(bboxes)
                bboxes = bboxes[bboxes[:, -1] == 2]  # cars only

                # calculate distance
                bb_hs = (bboxes[:, 3] * config.img_height) - (bboxes[:, 1] * config.img_height)  # height bbox
                est_dist = np.array([config.est_focall] * len(bb_hs)) * config.est_height / bb_hs

                # store predictions
                im_paths = np.concatenate((im_paths, np.repeat(path, len(bboxes))))
                xmins = np.concatenate((xmins, bboxes[:, 1]))
                ymins = np.concatenate((ymins, bboxes[:, 2]))
                xmaxs = np.concatenate((xmaxs, bboxes[:, 3]))
                ymaxs = np.concatenate((ymaxs, bboxes[:, 4]))
                confs = np.concatenate((confs, bboxes[:, 4]))
                dists = np.concatenate((dists, est_dist))

                ######### compute f1-score
                target = torch.zeros((config.row_size, self.num_predictors)).to(self.device)
                if has_labels:
                    for label in pred_labels:
                        angle_i = int(label[0] // self.cell_angle)
                        distance_i = int(label[1] // self.cell_depth)
                        target[angle_i, distance_i] = 1.0

            # update tqdm progress bar
            pbar.update(images.shape[0])

    # save into dataframe
    data = {
        "path": im_paths,
        "xmin": xmins,
        "ymin": ymins,
        "xmax": xmaxs,
        "ymax": ymaxs,
        "dist": dists,
        "conf": confs,
    }
    df = pd.DataFrame(data)
    df.to_csv("scripts/yolo_predictions.csv", index=False)


def get_args(Cfg):
    # pars arguments
    parser = argparse.ArgumentParser(description="Use Yolov4 to predict on a nuScenes split")
    parser.add_argument(
        "-s", "--split", type=str, default="test", help="split to use in nuScenes dataset", dest="split"
    )
    parser.add_argument(
        "-n", "--names_file", type=str, default="names/coco.names", help="names directory", dest="names_path"
    )
    parser.add_argument(
        "-d_dir",
        "--dataset_dir",
        type=str,
        default="../data/nuScenes/splits",
        help="path to nuScenes directory",
        dest="dataset_dir",
    )

    args = vars(parser.parse_args())
    Cfg.update(args)

    return Cfg


if __name__ == "__main__":

    Cfg = get_args(Cfg)

    # look for available devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    Cfg.device = device

    model = Darknet(Cfg.cfgfile, model_type="Yolov4")
    model.load_weights(Cfg.weights_file)

    # get num parameters
    print(f"model_params = {model.num_params()}")

    # load model and push to device
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    predict_and_save(model, device, Cfg)


# Height mean:        1.6843582080346167
# Focal length mean:  1532.27214617596
# Error mean:         0.806496207113689