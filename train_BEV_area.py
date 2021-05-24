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

from dataset import Yolo_BEV_dataset
from cfg.train.cfg_yolov4_BEV_area_nuScenes_max import Cfg

from tool.darknet2pytorch import Darknet


def collate(batch):
    images = []
    labels = []
    for img, lab in batch:
        images.append([img])
        if len(lab.shape) == 1:
            labels.append(np.expand_dims(lab, 0))
        else:
            labels.append(lab)

    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    return images, labels


class Yolo_loss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.loss = cfg.loss

        # focal loss
        self.gamma = 6.0
        self.alpha = 50.0

        # params
        self.device = device
        self.cell_angle = cfg.cell_angle
        self.cell_depth = cfg.cell_depth
        self.num_predictors = cfg.num_predictors

    def forward(self, preds, labels, eval=False):
        # params
        loss = 0.0
        row_size = preds.shape[-1] if not eval else preds.shape[1]

        for pred_id, pred in enumerate(preds):

            ################### get pred
            if not eval:
                pred = pred.view(self.num_predictors, row_size)
                pred = pred.permute(1, 0).contiguous()

            # GPU fix
            pred = torch.clone(pred)

            # transform pred
            if not eval:
                pred = torch.sigmoid(pred)

            ################### get labels
            target = torch.Tensor(labels[pred_id]).float().to(self.device)

            ################### compute loss
            if self.loss == "focal_loss":
                pt = torch.where(target == 1.0, pred, 1.0 - pred)
                ce = -torch.log(pt)
                alpha_ = torch.where(target == 1.0, self.alpha, 1.0)
                focal_loss = alpha_ * ((1.0 - pt) ** self.gamma) * ce
                loss += focal_loss.sum()

            elif self.loss == "dice_loss":
                numerator = 2 * (pred * target).sum()
                denominator = (pred ** 2).sum() + (target ** 2).sum() + 1e-16
                dice_loss = 1 - numerator / denominator
                loss += dice_loss

        return loss


def train(
    model,
    device,
    config,
    epochs=5,
    save_cp=True,
    log_step=20,
):
    # Get dataloaders
    train_dataset = Yolo_BEV_dataset(config, split="train", input_type="nuScenes", return_area=True)
    val_dataset = Yolo_BEV_dataset(config, split="val", input_type="nuScenes", return_area=True)

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
        Grid dimension:  {config.num_columns}x{config.num_predictors}
        Loss function:   {config.loss}
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
    for epoch in range(epochs):
        epoch_loss = 0

        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img", ncols=100) as pbar:

            ################ training
            batch_loss = 0
            for i, batch in enumerate(train_loader):
                # get batch
                global_step += 1
                images = batch[0].float().to(device)
                labels = batch[1]

                # compute loss
                preds = model(images)[0]
                batch_loss += criterion(preds, labels)

                # update weights
                if global_step % config.subdivisions == 0:
                    # set losses
                    epoch_loss += batch_loss.item()
                    batch_loss /= config.batch

                    # compute gradients
                    batch_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    # batch log
                    if global_step % (log_step * config.subdivisions) == 0:
                        writer.add_scalar("train/Loss", batch_loss.item(), global_step)
                        writer.add_scalar("lr", scheduler.get_lr()[0] * config.batch, global_step)

                    batch_loss = 0

                pbar.update(images.shape[0])

            # epoch log
            writer.add_scalar("Epoch/loss", epoch_loss / n_train, epoch)

            ################ validation
            # evaluate model
            if epoch % 2 == 0:

                # get evaluation model
                eval_model = Darknet(cfg.cfgfile, model_type="BEV_dist")
                if torch.cuda.device_count() > 1:
                    eval_model = torch.nn.DataParallel(eval_model)
                eval_model.load_state_dict(model.state_dict())
                eval_model.to(device)
                eval_model.eval()

                # evaluation loop
                with torch.no_grad():
                    eval_loss = 0.0
                    for batch in tqdm(val_loader, desc="Val", leave=False):

                        # get batch
                        global_step += 1
                        images = batch[0].float().to(device)
                        labels = batch[1]

                        # compute loss
                        preds = eval_model(images)
                        loss = criterion(preds, labels, eval=True)
                        eval_loss += loss.item()

                    # log
                    writer.add_scalar("val/Loss", eval_loss / n_val, epoch)

                    del eval_model

                    # save checkpoint
                    if save_cp and ((eval_loss < min_eval_loss) or epoch % 10 == 0):
                        try:
                            os.makedirs(config.checkpoints, exist_ok=True)
                        except OSError:
                            pass

                        # decide name
                        if eval_loss < min_eval_loss:
                            min_eval_loss = eval_loss
                            save_path = os.path.join(
                                config.checkpoints,
                                f"{config.save_prefix}_epoch{epoch + 1}_{config.save_postfix}_BESTSOFAR.pth",
                            )
                        else:
                            save_path = os.path.join(
                                config.checkpoints,
                                f"{config.save_prefix}_epoch{epoch + 1}_{config.save_postfix}.pth",
                            )
                            saved_models.append(save_path)

                        # save model
                        torch.save(model.state_dict(), save_path)
                        logging.info(f"Checkpoint {epoch + 1} saved !")
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
        default="Yolo_BEV_area_nuScenes",
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
        default=85,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    cfg.device = device

    model = Darknet(cfg.cfgfile, model_type="BEV_dist")

    # load weights
    if cfg.load.endswith(".weights"):
        model.load_weights(cfg.load, cut_off=85)
    else:
        model.load_state_dict(torch.load(cfg.load))

    # freeze backbone
    if cfg.freeze:
        model.freeze_layers([i for i in range(cfg.freeze)])
    else:
        print("!! Training Backbone !!")

    # get num parameters
    print(f"model_params = {model.num_params()}")

    # load model and push to device
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

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
