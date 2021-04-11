import os
from easydict import EasyDict

_BASE_DIR = os.path.abspath(".")

Cfg = EasyDict()

Cfg.cfgfile = os.path.join(_BASE_DIR, "cfg", "model", "yolov4_BEV_flat.cfg")

Cfg.epochs = 50
Cfg.batch = 64
Cfg.subdivisions = 32
Cfg.width = 200
Cfg.height = 136
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = 0.1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = 0.1, 0.1

# BEV params
Cfg.num_predictors = 20
Cfg.cell_angle = 3.33
Cfg.cell_depth = 5
Cfg.anchors = 1.64, 3.93
Cfg.num_anchors = 1

# data augmentation
Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 80
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_OPTIMIZER = "adam"

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, "log")

Cfg.iou_type = "rgIoU"  # ['IoU', 'gIoU', 'rIoU', 'gIoU']

Cfg.keep_checkpoint_max = 10
