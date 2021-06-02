import logging
import os, sys, math
import argparse
from collections import deque
import datetime

from torch.nn.modules import module
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
from tool.utils import my_IoU

from dataset import Yolo_BEV_dataset
from cfg.train.cfg_yolov4_BEV_flat_nuScenes import Cfg
import pandas as pd

from tool.darknet2pytorch import Darknet


def collate(batch):
    images = []
    labels = []
    paths = []
    for img, lab, path in batch:
        images.append([img])
        paths.append([path])
        if len(lab.shape) == 1:
            labels.append(np.expand_dims(lab, 0))
        else:
            labels.append(lab)

    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    return images, labels, paths


class Yolo_loss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        # losses
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCELoss(reduction="sum")
        # self.sl1 = nn.SmoothL1Loss(reduction="sum")

        # constants
        self.lambda_xy = 10
        self.lambda_wl = 10
        self.lambda_rot = 20
        self.lambda_obj = 20
        self.lambda_noobj = 1

        # self.alpha = 0.25
        # self.gamma = 2

        # params
        self.device = device
        self.cell_angle = cfg.cell_angle
        self.cell_depth = cfg.cell_depth
        self.bbox_attrib = 7
        self.num_predictors = cfg.num_predictors
        self.num_anchors = cfg.num_anchors
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

            # GPU fix
            pred = torch.clone(pred)

            # transform pred
            pred[..., :2] = torch.sigmoid(pred[..., :2])  # xy
            pred[..., 2:4] = pred[..., 2:4]  # wl better gradient flow if we transf target rather than pred
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
            # loss_wl += self.mse(torch.sqrt(pred[obj_mask][..., 2:4]), torch.sqrt(target[obj_mask][..., 2:4]))

            ####### rotation loss
            loss_rot += self.mse(pred[obj_mask][..., 4:6], target[obj_mask][..., 4:6])

            ####### obj loss (same as class loss)
            loss_obj += self.bce(pred[obj_mask][..., 6:], target[obj_mask][..., 6:])

            ####### noobj loss
            loss_noobj += self.bce(pred[noobj_mask][..., 6:], target[noobj_mask][..., 6:])

            ###### box loss
            # target[..., 2:3] = torch.log(target[..., 2:3] / self.anchors[0][0] + 1e-16)  # w
            # target[..., 3:4] = torch.log(target[..., 3:4] / self.anchors[0][1] + 1e-16)  # l
            # loss_box += self.sl1(pred[..., :6], target[..., :6])

            # ####### focal loss (obj and noobj)
            # ce = self.bce(pred[..., 6:], target[..., 6:])
            # alpha = target[..., 6:] * self.alpha + (1.0 - target[..., 6:]) * (1.0 - self.alpha)
            # pt = torch.where(target[..., 6:] == 1, pred[..., 6:], 1 - pred[..., 6:])
            # loss_focal += alpha * (1.0 - pt) ** self.gamma * ce

        loss = (
            self.lambda_xy * loss_xy
            + self.lambda_wl * loss_wl
            + self.lambda_rot * loss_rot
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
        )

        return loss, loss_xy, loss_wl, loss_rot, loss_obj, loss_noobj


def compute_AP(df_preds, n_gt):
    df_preds.sort_values("conf", ascending=False, inplace=True)

    true_sum = 0
    precision = []
    recall = []
    for i, value in enumerate(df_preds["correct"]):
        true_sum += value
        precision.append(true_sum / (i + 1))
        recall.append(true_sum / n_gt)

    # precision interpolation
    for i in range(len(precision)):
        if i < len(precision) - 1:
            precision[i] = max(precision[i + 1 :])
    precision = [1.0] + precision + [0.0, 0.0]
    recall = [0.0] + recall + [recall[-1] + 1e-16, 1.0]

    # computing interpolated AP (VOC format)
    ap_voc = 0
    prev_i, prev_prec = 0, precision[0]
    for i, prec in enumerate(precision[1:]):
        if prec < prev_prec:
            ap_voc += (recall[i] - recall[prev_i]) * prev_prec
            prev_i = i
            prev_prec = prec

    return ap_voc


def train(
    model,
    device,
    config,
    epochs=5,
    save_cp=True,
    log_step=20,
):
    # Get dataloaders
    train_dataset = Yolo_BEV_dataset(config, split="train", input_type="nuScenes")
    val_dataset = Yolo_BEV_dataset(config, split="val", input_type="nuScenes")

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
        filename_suffix=f"OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}_{cfg.save_postfix}",
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
    saved_models = deque()
    model.train()
    min_eval_loss = math.inf
    max_AP = 0.0
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img", ncols=100) as pbar:
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
                    logging.debug(
                        "Train step_{} -> loss : {}, loss xy : {}, loss wl : {}, "
                        "loss rot : {}，loss obj : {}, loss noobj : {}, lr : {}".format(
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

            # epochs log
            writer.add_scalar("Epoch/loss", epoch_loss, epoch)
            writer.add_scalar("Epoch/mean_loss", epoch_loss / n_train, epoch)

            # evaluate models
            with torch.no_grad():
                if epoch % 2 == 0:
                    eval_model_loss = Darknet(cfg.cfgfile, model_type="BEV_flat")
                    eval_model_AP = Darknet(cfg.cfgfile, inference=True, model_type="BEV_flat")
                    if torch.cuda.device_count() > 1:
                        eval_model_loss.load_state_dict(model.module.state_dict())
                        eval_model_AP.load_state_dict(model.module.state_dict())
                    else:
                        eval_model_loss.load_state_dict(model.state_dict())
                        eval_model_AP.load_state_dict(model.state_dict())
                    eval_model_loss.to(device)
                    eval_model_AP.to(device)
                    eval_model_AP.eval()

                    eval_loss = 0.0
                    eval_loss_xy = 0.0
                    eval_loss_wl = 0.0
                    eval_loss_rot = 0.0
                    eval_loss_obj = 0.0
                    eval_loss_noobj = 0.0

                    print("\nEvaluating...")
                    n_gt = 0
                    df_preds = pd.DataFrame(columns=["conf", "correct"])
                    for batch in tqdm(val_loader, desc="Batch", leave=False):

                        # get batch
                        global_step += 1
                        epoch_step += 1
                        images = batch[0].float().to(device=device)
                        labels = batch[1]

                        # compute loss
                        labels_pred = eval_model_loss(images)[0].detach()
                        loss, loss_xy, loss_wl, loss_rot, loss_obj, loss_noobj = criterion(
                            labels_pred, labels
                        )
                        eval_loss += loss.item()
                        eval_loss_xy += loss_xy.item()
                        eval_loss_wl += loss_wl.item()
                        eval_loss_rot += loss_rot.item()
                        eval_loss_obj += loss_obj.item()
                        eval_loss_noobj += loss_noobj.item()

                        # TODO: fix this double inference
                        # save tuple to compute AP
                        preds_AP = eval_model_AP(images).detach()

                        for pred_id, preds in enumerate(preds_AP):
                            label = labels[pred_id]
                            n_gt = n_gt if label[0][-1] == -1 else n_gt + len(label)

                            mask_conf = (
                                torch.nonzero((preds[:, -1] >= cfg.conf_thresh).float()).squeeze().to(device)
                            )
                            preds = preds[mask_conf]
                            if preds.shape[0] == 0:
                                continue
                            if len(preds.shape) == 1:
                                preds = preds.unsqueeze(0)

                            if label[0][-1] == -1:
                                for pred in preds:
                                    df_preds = df_preds.append(
                                        pd.Series({"conf": pred[-1].item(), "correct": 0.0}),
                                        ignore_index=True,
                                    )
                            else:
                                for pred in preds:
                                    try:
                                        iou_scores = my_IoU(
                                            pred.unsqueeze(0),
                                            torch.Tensor(label),
                                            cfg.iou_type,
                                            cfg.device,
                                        )
                                    except Exception:
                                        break

                                    correct = (iou_scores >= cfg.iou_thresh).any().float().item()
                                    df_preds = df_preds.append(
                                        pd.Series({"conf": pred[-1].item(), "correct": correct}),
                                        ignore_index=True,
                                    )

                    # compute AP
                    df_preds.astype({"conf": float, "correct": float})
                    val_AP = compute_AP(df_preds, n_gt)

                    # log
                    logging.debug(
                        "Val step_{} -> loss : {}, loss xy : {}, loss wl : {},"
                        " loss rot : {}，loss obj : {}, loss noobj : {}, lr : {}".format(
                            global_step,
                            eval_loss,
                            eval_loss_xy,
                            eval_loss_wl,
                            eval_loss_rot,
                            eval_loss_obj,
                            eval_loss_noobj,
                            scheduler.get_lr()[0] * config.batch,
                        )
                    )
                    writer.add_scalar("val/Loss", eval_loss, epoch)
                    writer.add_scalar("val/loss_xy", eval_loss_xy, epoch)
                    writer.add_scalar("val/loss_wl", eval_loss_wl, epoch)
                    writer.add_scalar("val/loss_rot", eval_loss_rot, epoch)
                    writer.add_scalar("val/loss_obj", eval_loss_obj, epoch)
                    writer.add_scalar("val/loss_noobj", eval_loss_noobj, epoch)
                    writer.add_scalar("val/AP", val_AP, epoch)

                    del eval_model_loss, eval_model_AP, df_preds

                    # save checkpoint
                    if save_cp and ((eval_loss < min_eval_loss and val_AP > max_AP) or epoch % 10 == 0):
                        max_AP = val_AP
                        min_eval_loss = eval_loss
                        try:
                            os.makedirs(config.checkpoints, exist_ok=True)
                            logging.info("Created checkpoint directory")
                        except OSError:
                            pass
                        save_path = os.path.join(
                            config.checkpoints, f"{cfg.save_prefix}_epoch{epoch + 1}_{cfg.save_postfix}.pth"
                        )
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
        "-spost",
        "--save_postfix",
        metavar="save_postfix",
        type=str,
        nargs="?",
        default="",
        help="postfix for saved files' names",
        dest="save_postfix",
    )
    parser.add_argument(
        "-spre",
        "--save_prefix",
        metavar="save_prefix",
        type=str,
        nargs="?",
        default="Yolo_BEV_flat_nuScenes",
        help="prefix for saved files' names",
        dest="save_prefix",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.001,
        help="Learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "-l",
        "--load",
        dest="load",
        type=str,
        default=os.path.abspath(".") + "/checkpoints/yolov4.weights",
        help="Path to a .pth file to load",
    )
    parser.add_argument("-g", "--gpu", metavar="G", type=str, default="-1", help="GPU id", dest="gpu")
    parser.add_argument(
        "-dir",
        "--data-dir",
        type=str,
        default=os.path.abspath("..") + "/data/nuScenes/splits",
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
        "-f",
        "--freeze-backbone",
        type=int,
        default=54,
        help="number of layers to freeze",
        dest="freeze",
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
    if cfg.load.endswith(".weights"):
        model.load_weights(cfg.load, cut_off=53)
    else:
        model.load_state_dict(torch.load(cfg.load))

    # freeze backbone
    if cfg.freeze:
        model.freeze_layers([i for i in range(cfg.freeze)])
    else:
        print("!! Training Backbone !!")

    # get num parameters
    print(f"model_params = {model.num_params()}")

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
