import os
from easydict import EasyDict

_BASE_DIR = os.path.abspath(".")

Cfg = EasyDict()

# paths
Cfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, "log")
Cfg.cfgfile = os.path.join(_BASE_DIR, "cfg", "model", "yolov4_BEV_area_nuScenes_max.cfg")

# nuScenes input dim
Cfg.width = 240
Cfg.height = 136
Cfg.channels = 3

# nuScenes BEV params
Cfg.cell_angle = 4.6666666667
Cfg.cell_depth = 2.5
Cfg.num_predictors = 20
Cfg.num_columns = 15

# training params
Cfg.epochs = 100
Cfg.batch = 64
Cfg.subdivisions = 4
Cfg.max_batches = 6000000

# loss and optimizer
Cfg.loss = "focal_loss"
Cfg.TRAIN_OPTIMIZER = "sgd"

# learning rate
Cfg.learning_rate = 0.00261
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.burn_in = 10000
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = 0.1, 0.1

# extra
Cfg.keep_checkpoint_max = 5
