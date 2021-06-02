import torch
from tool.darknet2pytorch import Darknet
from torch.utils.data import DataLoader
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import Yolo_BEV_dataset

# params
is_area = False

if is_area:
    from cfg.train.cfg_yolov4_BEV_area_nuScenes import Cfg as cfg
else:
    from cfg.train.cfg_yolov4_BEV_dist_nuScenes import Cfg as cfg

cfg.weights = (
    "./checkpoints/dist_focalloss_noalpha_gamma6_rows10_sgd/Yolo_BEV_dist_nuScenes_epoch43__BESTSOFAR.pth"
)
cfg.dataset_dir = "../data/nuScenes/splits"
cfg.names_path = "./names/BEV.names"
cfg.save_predictions = False

# check for cuda
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# collate function for data loader
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


if __name__ == "__main__":

    tresholds = np.arange(0, 1, 0.01)
    stats = {t: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for t in tresholds}

    # dataset
    test_dataset = Yolo_BEV_dataset(cfg, split="val", input_type="nuScenes", return_area=is_area)
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
    m.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        m = torch.nn.DataParallel(m)
    m.to(device)
    m.eval()

    # start testing
    n_test = len(test_dataset)
    with tqdm(total=n_test, unit="img", ncols=100) as pbar:
        for batch_id, batch in enumerate(test_loader):
            images = batch[0].float().to(device)
            labels = batch[1]

            preds = m(images)

            row_size = preds.shape[1]
            for pred_id, (pred, img) in enumerate(zip(preds, images)):

                if is_area:
                    target = torch.Tensor(labels[pred_id]).float().to(device)
                else:
                    # generate labels
                    pred_labels = labels[pred_id]
                    pred_labels[:, 0] += (row_size * cfg.cell_angle) / 2  # adjust angle with fov
                    has_labels = pred_labels[0][-1] >= 0.0

                    # target matrix
                    target = torch.zeros((row_size, cfg.num_predictors)).to(device)
                    if has_labels:
                        for label in pred_labels:
                            angle_i = int(label[0] // cfg.cell_angle)
                            distance_i = int(label[1] // cfg.cell_depth)
                            target[angle_i, distance_i] = 1.0

                # compute stats
                for t in stats.keys():
                    pred_tmp = torch.where(pred >= t, 1.0, 0.0)
                    stats[t]["TP"] += torch.where((target == 1.0) & (pred_tmp == 1.0), 1.0, 0.0).sum().item()
                    stats[t]["TN"] += torch.where((target == 0.0) & (pred_tmp == 0.0), 1.0, 0.0).sum().item()
                    stats[t]["FP"] += torch.where((target == 0.0) & (pred_tmp == 1.0), 1.0, 0.0).sum().item()
                    stats[t]["FN"] += torch.where((target == 1.0) & (pred_tmp == 0.0), 1.0, 0.0).sum().item()

                # save preds
                if cfg.save_predictions:
                    if not os.path.exists("predictions"):
                        os.makedirs("predictions/images")
                        os.makedirs("predictions/preds")
                        os.makedirs("predictions/labels")

                    target = target.detach().cpu().numpy().transpose()
                    pred = pred.detach().cpu().numpy().transpose()
                    img = img.detach().cpu().numpy().transpose(1, 2, 0)

                    id = pred_id + (batch_id * len(images))
                    plt.imsave(f"predictions/images/{id}.png", img)
                    plt.imsave(f"predictions/preds/{id}.png", pred, origin="lower")
                    plt.imsave(f"predictions/labels/{id}.png", target, origin="lower")

            pbar.update(len(images))

    # compute stats
    max_pr, max_ac, max_re, max_f1, t_pr, t_ac, t_re, t_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for t in stats.keys():
        stats[t]["Precision"] = stats[t]["TP"] / (stats[t]["TP"] + stats[t]["FP"] + 1e-16)
        stats[t]["Accuracy"] = (stats[t]["TP"] + stats[t]["TN"]) / (
            stats[t]["TP"] + stats[t]["FP"] + stats[t]["TN"] + stats[t]["FN"] + 1e-16
        )
        stats[t]["Recall"] = stats[t]["TP"] / (stats[t]["TP"] + stats[t]["FN"] + 1e-16)
        stats[t]["F1-score"] = 2 * (
            (stats[t]["Precision"] * stats[t]["Recall"])
            / (stats[t]["Precision"] + stats[t]["Recall"] + 1e-16)
        )

        # get max
        max_pr, t_pr = (stats[t]["Precision"], t) if stats[t]["Precision"] > max_pr else (max_pr, t_pr)
        max_ac, t_ac = (stats[t]["Accuracy"], t) if stats[t]["Accuracy"] > max_ac else (max_ac, t_ac)
        max_re, t_re = (stats[t]["Recall"], t) if stats[t]["Recall"] > max_re else (max_re, t_re)
        max_f1, t_f1 = (stats[t]["F1-score"], t) if stats[t]["F1-score"] > max_f1 else (max_f1, t_f1)

    with open("stats.json", "w") as f:
        json.dump(stats, f)

    # results
    print("full results saved into stats.json file")
    print(f"max precision {max_pr} with treshold {t_pr}")
    print(f"max accuracy {max_ac} with treshold {t_ac}")
    print(f"max recall {max_re} with treshold {t_re}")
    print(f"max f1-score {max_f1} with treshold {t_f1}")