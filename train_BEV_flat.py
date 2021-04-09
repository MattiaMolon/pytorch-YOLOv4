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


class Yolo_loss_OLD(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()

        # general hyperparams
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        # anchors
        self.anchors = [
            [12, 16],
            [19, 36],
            [40, 28],
            [36, 75],
            [76, 55],
            [72, 146],
            [142, 110],
            [192, 243],
            [459, 401],
        ]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        (
            self.masked_anchors,
            self.ref_anchors,
            self.grid_x,
            self.grid_y,
            self.anchor_w,
            self.anchor_h,
        ) = ([], [], [], [], [], [])

        # anchors adaptation to grid size
        for i in range(3):
            # scale to grid dimension
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]

            # take only anchors related to this grid dimension
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)

            # reference anchors of shape(x=0,y=0,w,h) where w and h are resized to grid
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)

            # define grid of indexes (Cx, Cy)
            fsize = image_size // self.strides[i]
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            grid_y = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(batch, 3, fsize, 1)
                .permute(0, 1, 3, 2)
                .to(device)
            )

            # repeat anchors w and h for each cell
            anchor_w = (
                torch.from_numpy(masked_anchors[:, 0])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            anchor_h = (
                torch.from_numpy(masked_anchors[:, 1])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(
            device=self.device
        )
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # count number of objects in ground truth
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # adapt label to feature map dimension
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()  # int version(?)
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        # for each image in the batch
        for b in range(batchsize):

            # look how many labels are in the image
            n = int(nlabel[b])
            if n == 0:
                continue

            # get truth boxes
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            # return a list NxK where N is the number of bboxes in truth and K is the number of anchors
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)

            # find best anchor boxes compared to the truth
            best_n_all = anchor_ious_all.argmax(dim=1)  # best anchors for each truth
            best_n = best_n_all % 3
            best_n_mask = (
                (best_n_all == self.anch_masks[output_id][0])
                | (best_n_all == self.anch_masks[output_id][1])
                | (best_n_all == self.anch_masks[output_id][2])
            )  # control if there is at least 1 anchor of the base ones in the best iou

            # if the predictions are not big or small enough for this stride
            if sum(best_n_mask) == 0:
                continue

            # add x,y to truth bboxes so now they are [x,y,w,h] in feature maps dims
            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # compute iou between truth and predicted
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_thre
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])  # mask over the grids

            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~pred_best_iou

            # loop through truth bboxes
            for ti in range(best_n.shape[0]):
                # if we are in the right stride
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]  # anchor index for this layer
                    obj_mask[b, a, j, i] = 1  # where the bboxes are
                    tgt_mask[b, a, j, i, :] = 1  # ???

                    # define the target parameters
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(
                        torch.float
                    )  # offset on cell over x
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(
                        torch.float
                    )  # offset on cell over y
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0]
                        + 1e-16
                    )  # ??
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1]
                        + 1e-16
                    )  # ??
                    target[b, a, j, i, 4] = 1  # objectivness
                    target[
                        b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()
                    ] = 1  # one hot encoding

                    # target scale ??
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize
                    )

        # obj_mask  -> 1 where predictions are below ignore threshold of iou with gt
        #              or if ground truth belong to this stride 0 otherwise
        # tgt_mask  -> 1 if gt should be detected in this stride level
        # tgt_scale -> ?? no idea -> tgt_scale[b, a, j, i, :] = torch.sqrt(
        #                                       2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        # target    -> ground truth trasnformed into stride space [b, n_acnhors, w, h, n_ch]
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin: torch.Tensor, labels: torch.Tensor = None) -> Tuple:
        """Compute loss

        Args:
            xin (torch.Tensor): Network prediction. [boxes_stridex8, boxes_stridex16, boxes_predx32]
            labels (torch.Tensor, optional): Ground truth. Defaults to None.

        Returns:
            Tuple: Computed losses. (loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2)
        """
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0

        # loop over network outputs
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            # output [batch, n_anch*attrib, H, W] -> [batch, n_anch, H, W, attrib]
            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            # extract bounding boxes bxy = sigma(txy) + Cxy, bwh = exp(twh) * pwh
            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            # obj_mask  -> 1 where predictions are below ignore threshold of iou with gt
            #              or if ground truth belong to this stride 0 otherwise
            # tgt_mask  -> 1 if gt should be detected in this stride level
            # tgt_scale -> ?? no idea -> tgt_scale[b, a, j, i, :] = torch.sqrt(
            #                                       2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
            # target    -> ground truth trasnformed into stride space [b, n_acnhors, w, h, n_ch]
            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
                pred, labels, batchsize, fsize, n_ch, output_id
            )

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(
                input=output[..., :2],
                target=target[..., :2],
                weight=tgt_scale * tgt_scale,
                reduction="sum",
            )
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction="sum") / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction="sum")
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction="sum")
            loss_l2 += F.mse_loss(input=output, target=target, reduction="sum")

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


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
        self.lambda_xy = 1
        self.lambda_wl = 1
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

        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img", ncols=50) as pbar:
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
                eval_model = Darknet(cfg.cfgfile, inference=True)
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
                with tqdm(total=n_val, desc=f"Eval {(epoch + 1) // 2}", unit="img", ncols=50) as epbar:
                    for i, batch in enumerate(val_loader):
                        # get batch
                        global_step += 1
                        epoch_step += 1
                        images = batch[0].float().to(device=device)
                        labels = batch[1]

                        # compute loss
                        labels_pred = model(images)
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
