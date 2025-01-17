import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from numpy.core.fromnumeric import size
from torch.nn.modules import module
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from dataset import Yolo_BEV_dataset
from cfg.train.cfg_yolov4_BEV_flat import Cfg

from tool.darknet2pytorch import Darknet
from tool.utils import my_IoU

from typing import Dict, Tuple, List


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        if len(box.shape) == 1:
            bboxes.append(np.expand_dims(box, 0))
        else:
            bboxes.append(box)

    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    return images, bboxes


class Yolo_loss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        # losses
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCELoss(reduction="sum")

        # constants
        self.lambda_xy = 10
        self.lambda_wl = 10
        self.lambda_rot = 10
        self.lambda_obj = 10
        self.lambda_noobj = 1

        # params
        self.device = device
        self.cell_angle = cfg.cell_angle
        self.cell_depth = cfg.cell_depth
        self.bbox_attrib = 7
        self.num_predictors = 20
        self.num_anchors = 1
        self.anchors = [(cfg.anchors[i], cfg.anchors[i + 1]) for i in range(0, len(cfg.anchors), 2)]

    def forward(self, preds, labels):
        # TODO: need to adapt in case of multiple anchors or multiple classes
        # params
        loss, loss_xy, loss_wl, loss_rot, loss_obj, loss_noobj = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        row_size = preds.shape[-1]

        for pred_id, pred in enumerate(preds):
            ################### get pred
            pred = pred.view(self.num_anchors * self.num_predictors * self.bbox_attrib, row_size)
            pred = pred.permute(1, 0).contiguous()
            pred = pred.view(row_size, self.num_predictors, self.num_anchors * self.bbox_attrib)

            # transform pred
            pred[..., :2] = torch.sigmoid(pred[..., :2])  # xy
            pred[..., 2:4] = pred[..., 2:4]  # wl better gradient flow if we tranf target rather than pred
            pred[..., 4:6] = torch.tanh(pred[..., 4:6])  # rotation
            pred[..., 6:7] = torch.sigmoid(pred[..., 6:7])  # confidence class

            ################### get labels
            pred_labels = labels[pred_id]
            pred_labels[:, 0] += (row_size * self.cell_angle) / 2  # adjust angle with fov
            has_labels = pred_labels[0][-1] >= 0.0

            # obj and no obj mask
            obj_mask = torch.zeros((row_size, self.num_predictors * self.num_anchors)).to(self.device)
            if has_labels:  # some gt is present
                for label in pred_labels:
                    i, j = int(label[0] // self.cell_angle), int(label[1] // self.cell_depth)
                    obj_mask[i, j] = 1.0
            obj_mask = obj_mask.bool()
            noobj_mask = (~obj_mask).to(self.device)

            # target matrix
            target = torch.zeros((row_size, self.num_predictors, self.bbox_attrib * self.num_anchors)).to(
                self.device
            )
            if has_labels:
                for label in pred_labels:
                    i, j = int(label[0] // self.cell_angle), int(label[1] // self.cell_depth)

                    # normalize labels
                    label[0] = label[0] / self.cell_angle - i  # x
                    label[1] = label[1] / self.cell_depth - j  # y
                    label[6] = 1.0

                    label = torch.from_numpy(label).to(device)
                    target[i, j] = label

            ################### compute losses
            ####### box coordinates loss
            loss_xy += self.mse(pred[obj_mask][..., :2], target[obj_mask][..., :2])
            target[..., 2:3] = torch.log(target[..., 2:3] / self.anchors[0][0] + 1e-16)  # w
            target[..., 3:4] = torch.log(target[..., 3:4] / self.anchors[0][1] + 1e-16)  # l
            loss_wl += self.mse(pred[obj_mask][..., 2:4], target[obj_mask][..., 2:4])
            # loss_wh += self.mse(torch.sqrt(pred[obj_mask][..., 2:4]), torch.sqrt(target[obj_mask][..., 2:4]))

            ####### rotation loss
            loss_rot += self.mse(pred[obj_mask][..., 4:6], target[obj_mask][..., 4:6])

            ####### obj loss (same as class loss)
            loss_obj += self.bce(pred[obj_mask][..., 6:], target[obj_mask][..., 6:])

            ####### noobj loss
            loss_noobj += self.bce(pred[noobj_mask][..., 6:], target[noobj_mask][..., 6:])

        loss = (
            self.lambda_xy * loss_xy
            + self.lambda_wl * loss_wl
            + self.lambda_rot * loss_rot
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
        )

        return loss, loss_xy, loss_wl, loss_rot, loss_obj, loss_noobj


def train(
    model,
    device,
    config,
    epochs=5,
    save_cp=True,
    log_step=20,
):
    # Get dataloaders
    train_dataset = Yolo_BEV_dataset(config, split="train")
    val_dataset = Yolo_BEV_dataset(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )

    # define summary writer
    writer = SummaryWriter(
        log_dir=config.TRAIN_TENSORBOARD_DIR,
        filename_suffix=f"OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}",
        comment=f"OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}",
    )

    # log
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    global_step = 0
    logging.info(
        f"""Starting training:
        Epochs:          {config.epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Input height:    {config.height}
        Input width:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
    """
    )

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    # optimizer + scheduler
    if config.TRAIN_OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )

    # scheduler multiplies learning rate by a factor calculated on epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    # loss function
    criterion = Yolo_loss(
        cfg=config,
        device=device,
    )

    # start training
    save_prefix = "Yolov4_BEV_flat_epoch"
    saved_models = deque()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img", ncols=75) as pbar:
            for i, batch in enumerate(train_loader):
                # get batch
                global_step += 1
                epoch_step += 1
                images = batch[0].float().to(device=device)
                labels = batch[1]

                # compute loss
                preds = model(images)[0]
                loss, loss_xy, loss_wl, loss_rot, loss_obj, loss_noobj = criterion(preds, labels)
                loss.backward()

                epoch_loss += loss.item()

                # update weights
                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                # log
                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar("train/Loss", loss.item(), global_step)
                    writer.add_scalar("train/loss_xy", loss_xy.item(), global_step)
                    writer.add_scalar("train/loss_wl", loss_wl.item(), global_step)
                    writer.add_scalar("train/loss_rot", loss_rot.item(), global_step)
                    writer.add_scalar("train/loss_obj", loss_obj.item(), global_step)
                    writer.add_scalar("train/loss_noobj", loss_noobj.item(), global_step)
                    writer.add_scalar("lr", scheduler.get_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(
                        {
                            "loss (batch)": loss.item(),
                            "loss_xy": loss_xy.item(),
                            "loss_wl": loss_wl.item(),
                            "loss_rot": loss_rot.item(),
                            "loss_obj": loss_obj.item(),
                            "loss_noobj": loss_noobj.item(),
                            "lr": scheduler.get_lr()[0] * config.batch,
                        }
                    )
                    logging.debug(
                        "Train step_{}: loss : {},loss xy : {},loss wl : {},"
                        "loss rot : {}，loss obj : {},loss noobj : {},lr : {}".format(
                            global_step,
                            loss.item(),
                            loss_xy.item(),
                            loss_wl.item(),
                            loss_rot.item(),
                            loss_obj.item(),
                            loss_noobj.item(),
                            scheduler.get_lr()[0] * config.batch,
                        )
                    )

                pbar.update(images.shape[0])

            # evaluate models
            min_eval_loss = math.inf
            if epoch % 2 == 0:
                eval_model = Darknet(cfg.cfgfile, inference=True, model_type="BEV_flat")
                if torch.cuda.device_count() > 1:
                    eval_model.load_state_dict(model.module.state_dict())
                else:
                    eval_model.load_state_dict(model.state_dict())
                eval_model.to(device)
                eval_model.eval()

                eval_loss = 0.0
                eval_loss_xy = 0.0
                eval_loss_wl = 0.0
                eval_loss_rot = 0.0
                eval_loss_obj = 0.0
                eval_loss_noobj = 0.0
                with tqdm(total=n_val, desc=f"Eval {(epoch + 1) // 2}", unit="img", ncols=75) as epbar:
                    for i, batch in enumerate(val_loader):
                        # get batch
                        global_step += 1
                        epoch_step += 1
                        images = batch[0].float().to(device=device)
                        labels = batch[1]

                        # compute loss
                        labels_pred = model(images)[0]
                        loss, loss_xy, loss_wl, loss_rot, loss_obj, loss_noobj = criterion(
                            labels_pred, labels
                        )
                        eval_loss += loss.item()
                        eval_loss_xy += loss_xy.item()
                        eval_loss_wl += loss_wl.item()
                        eval_loss_rot += loss_rot.item()
                        eval_loss_rot += loss_obj.item()
                        eval_loss_noobj += loss_noobj.item()

                        epbar.update(images.shape[0])

                # log
                logging.debug(
                    "Val step_{}: loss : {},loss xy : {},loss wl : {},"
                    "loss rot : {}，loss obj : {},loss noobj : {},lr : {}".format(
                        global_step,
                        eval_loss.item(),
                        eval_loss_xy.item(),
                        eval_loss_wl.item(),
                        eval_loss_rot.item(),
                        eval_loss_obj.item(),
                        eval_loss_noobj.item(),
                        scheduler.get_lr()[0] * config.batch,
                    )
                )

                del eval_model

            # save checkpoint
            if save_cp and eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                try:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f"{save_prefix}{epoch + 1}.pth")
                torch.save(model.state_dict(), save_path)
                logging.info(f"Checkpoint {epoch + 1} saved !")
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f"failed to remove {model_to_remove}")

    writer.close()


def get_args(**kwargs) -> dict:
    """Get args from command line and add them to cfg dictionary

    Returns:
        dict: dictionary with config params of the training and network
    """
    cfg = kwargs
    parser = argparse.ArgumentParser(
        description="Train the Model on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.001,
        help="Learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=None,
        help="Path to a .pth file to load",
    )
    parser.add_argument(
        "-b",
        "--backbone",
        dest="backbone",
        type=str,
        default=os.path.abspath(".") + "/checkpoints/yolov4.weights",
        help="Load model from a .pth file",
    )
    parser.add_argument("-g", "--gpu", metavar="G", type=str, default="-1", help="GPU id", dest="gpu")
    parser.add_argument(
        "-dir",
        "--data-dir",
        type=str,
        default=os.path.abspath("..") + "/data/KITTI/splits",
        help="dataset dir",
        dest="dataset_dir",
    )
    parser.add_argument(
        "-names",
        "--names-path",
        type=str,
        default=os.path.abspath(".") + "/names/BEV.names",
        help="names file location",
        dest="names_path",
    )
    parser.add_argument("-classes", type=int, default=1, help="dataset classes")
    parser.add_argument(
        "-optimizer",
        type=str,
        default="adam",
        help="training optimizer",
        dest="TRAIN_OPTIMIZER",
    )
    parser.add_argument(
        "-iou-type",
        type=str,
        default="IoU",
        help="iou type (IoU, gIoU, rIoU, rgIoU)",
        dest="iou_type",
    )
    parser.add_argument(
        "-keep-checkpoint-max",
        type=int,
        default=10,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    args = vars(parser.parse_args())

    cfg.update(args)

    return edict(cfg)


def init_logger(
    log_file: str = None,
    log_dir: str = None,
    log_level: int = logging.INFO,
    mode: str = "w",
    stdout: bool = True,
) -> module:
    """initialize logger

    Args:
        log_file (str, optional): log file name. Defaults to None.
        log_dir (str, optional): log directory. Defaults to None.
        log_level (int, optional): type of log level to use. Defaults to logging.INFO.
        mode (str, optional): mode of the logger, either w to overide or a to append. Defaults to "w".
        stdout (bool, optional): use stdout to print informations. Defaults to True.

    Returns:
        module: initialized logger
    """
    # format definition
    fmt = "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s"

    # logger directories
    if log_dir is None:
        log_dir = "~/temp/log/"
    if log_file is None:
        log_file = "log_" + get_date_str() + ".txt"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print("log file path:" + log_file)

    logging.basicConfig(level=logging.DEBUG, format=fmt, filename=log_file, filemode=mode)

    # use stdout
    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

    return logging


def get_date_str() -> str:
    """Get current date and time in a nice format "Y-M-D_h-m"

    Returns:
        str: string with date and time
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")


if __name__ == "__main__":

    # get logger and args
    logging = init_logger(log_dir="log")
    cfg = get_args(**Cfg)

    # look for available devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    cfg.device = device

    # load model and push to device
    model = Darknet(cfg.cfgfile, model_type="BEV_flat")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    # load weights
    if cfg.load is None:
        model.load_weights(cfg.backbone, cut_off=53)
    else:
        model.load_state_dict(cfg.load)

    # freeze backbone
    model.freeze_layers([i for i in range(54)])

    try:
        train(
            model=model,
            config=cfg,
            epochs=cfg.epochs,
            device=device,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "checkpoints/INTERRUPTED.pth")
        logging.info("Saved interrupt to checkpoints/INTERRUPTED.pth")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
