import os
from easydict import EasyDict

_BASE_DIR = os.path.abspath(".")

Cfg = EasyDict()

# paths
Cfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, "log")
Cfg.cfgfile = os.path.join(_BASE_DIR, "cfg", "model", "yolov4_BEV_flat_KITTI_canvas.cfg")

# input dim
Cfg.width = 200
Cfg.height = 136
Cfg.channels = 3

# training params
Cfg.epochs = 300
Cfg.batch = 64
Cfg.subdivisions = 16
Cfg.max_batches = 500500

# optimizer
Cfg.TRAIN_OPTIMIZER = "adam"

# learning rate
Cfg.learning_rate = 0.00261
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.burn_in = 1000
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = 0.1, 0.1

# BEV params
Cfg.cell_angle = 3.33
Cfg.cell_depth = 5
Cfg.anchors = 1.64, 3.93
Cfg.num_anchors = 1
Cfg.num_predictors = 20

# validation
Cfg.iou_type = "rIoU"  # ['IoU', 'gIoU', 'rIoU', 'gIoU']
Cfg.conf_thresh = 0.6
Cfg.iou_thresh = 0.1

# extra
Cfg.keep_checkpoint_max = 5
