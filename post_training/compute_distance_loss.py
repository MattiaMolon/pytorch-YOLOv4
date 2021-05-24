import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import torch
from tool.darknet2pytorch import Darknet
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import Yolo_BEV_dataset
from cfg.train.cfg_yolov4_BEV_area_nuScenes import Cfg as cfg
from torch.nn import MSELoss

cfg.weights = (
    "../checkpoints/area_focalloss_alpha30_gamma5_rows20_sgd/Yolo_BEV_area_nuScenes_epoch35__BESTSOFAR.pth"
)
cfg.dataset_dir = "../../data/nuScenes/splits"
cfg.names_path = "../names/BEV.names"
cfg.cfgfile = "../cfg/model/yolov4_BEV_area_nuScenes.cfg"
thresholds = 0.60

# check for cuda
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# collate function for data loader
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


# dataset
test_dataset = Yolo_BEV_dataset(cfg, split="val", input_type="nuScenes", return_area=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch // cfg.subdivisions,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate,
)

# remove "model" bug from parallel GPU training
state_dict = torch.load(cfg.weights, map_location=torch.device(device))
if "module" in list(state_dict.keys())[0]:
    state_dict_tmp = {}
    for k in state_dict.keys():
        state_dict_tmp[k[7:]] = state_dict[k]
    state_dict = state_dict_tmp

# model
m = Darknet(cfg.cfgfile, model_type="BEV_dist")
n_rows = int(m.blocks[-2]["filters"])
m.load_state_dict(state_dict)
if torch.cuda.device_count() > 1:
    m = torch.nn.DataParallel(m)
m.to(device)
m.eval()


# start predicting
loss_matrix = torch.zeros((n_rows, 1)).to(device)
n_test = len(test_dataset)
with tqdm(total=n_test, unit="img", ncols=100) as pbar:
    for batch_id, batch in enumerate(test_loader):
        images = batch[0].float().to(device)
        labels = batch[1]

        preds = m(images)

        # compute loss
        row_size = preds.shape[1]
        for pred_id, (pred, img) in enumerate(zip(preds, images)):

            target = torch.Tensor(labels[pred_id]).float().to(device).transpose(1, 0)
            pred_thresh = torch.where(pred >= thresholds, 1.0, 0.0).to(device).transpose(1, 0)

            loss_vector = []
            for t, pt, p in zip(target, pred_thresh, pred.transpose(1, 0)):
                loss_vector.append(MSELoss()(p, t))

            loss_vector = torch.Tensor(loss_vector).unsqueeze(-1).to(device)
            loss_matrix = torch.cat((loss_matrix, loss_vector), 1)

        pbar.update(len(images))


loss_matrix = loss_matrix[:, 1:].mean(1)
freq = torch.Tensor(
    [
        18.84164033,
        30.54564315,
        24.17569787,
        18.63670886,
        17.71720818,
        16.52413019,
        14.96239837,
        16.59864713,
        16.0731441,
        15.02346939,
        14.85671039,
        16.34073252,
        14.64975124,
        19.0465718,
        19.76241611,
        21.24531025,
        20.27961433,
        26.43267504,
        28.58834951,
        33.23476298,
    ]
).to(device)
loss_matrix = loss_matrix * freq
x = [(i + 1) * 2.5 for i in range(len(loss_matrix))]
plt.plot(x, loss_matrix.cpu().numpy())
plt.savefig("distance_loss.png")