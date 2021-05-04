import torch
from tool.darknet2pytorch import Darknet
from torch.utils.data import DataLoader
import json
import numpy as np
from tqdm import tqdm
from dataset import Yolo_BEV_dataset
from cfg.train.cfg_yolov4_BEV_dist_nuScenes import Cfg as cfg

# check for cuda
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# collate function for data loader
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


# params
cfg.weights = "./checkpoints/Yolo_BEV_dist_nuScenes_epoch81_.pth"
cfg.dataset_dir = "../data/nuScenes/splits"
cfg.model_file = "./cfg/model/yolov4_BEV_dist_nuScenes.cfg"
cfg.names_path = "./names/BEV.names"

if __name__ == "__main__":

    tresholds = [x * 0.05 for x in range(0, 20)][1:]
    stats = {t: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for t in tresholds}

    # dataset
    test_dataset = Yolo_BEV_dataset(cfg, split="train", input_type="nuScenes")
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch // cfg.subdivisions,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )

    # model
    m = Darknet(cfg.cfgfile, model_type="BEV_dist", inference=True).to(device)
    m.load_state_dict(torch.load(cfg.weights, map_location=torch.device(device)))
    m.eval()

    # start testing
    n_test = len(test_dataset)
    with tqdm(total=n_test, unit="img", ncols=100) as pbar:
        for batch in test_loader:
            images = batch[0].float().to(device)
            labels = batch[1]

            preds = m(images)

            row_size = preds.shape[1]
            for pred_id, pred in enumerate(preds):

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

                for t in stats.keys():
                    pred_tmp = torch.where(pred >= t, 1.0, 0.0)
                    stats[t]["TP"] += torch.where(target == 1 & pred_tmp == 1, 1.0, 0.0).sum()
                    stats[t]["TN"] += torch.where(target == 0 & pred_tmp == 0, 1.0, 0.0).sum()
                    stats[t]["FP"] += torch.where(target == 0 & pred_tmp == 1, 1.0, 0.0).sum()
                    stats[t]["FN"] += torch.where(target == 1 & pred_tmp == 0, 1.0, 0.0).sum()

    # compute stats
    max_pr, max_ac, max_re, max_f1 = 0.0, 0.0, 0.0, 0.0
    for t in stats.keys():
        stats[t]["Precision"] = stats[t]["TP"] / (stats[t]["TP"] + stats[t]["FP"])
        stats[t]["Accuracy"] = (stats[t]["TP"] + stats[t]["TN"]) / (
            stats[t]["TP"] + stats[t]["FP"] + stats[t]["TN"] + stats[t]["FN"]
        )
        stats[t]["Recall"] = stats[t]["TP"] / (stats[t]["TP"] + stats[t]["FN"])
        stats[t]["F1-score"] = 2 * (
            (stats[t]["Precision"] * stats[t]["Recall"]) / (stats[t]["Precision"] + stats[t]["Recall"])
        )

        # get max
        max_pr = stats[t]["precision"] if stats[t]["precision"] > max_pr else max_pr
        max_ac = stats[t]["Accuracy"] if stats[t]["Accuracy"] > max_ac else max_ac
        max_re = stats[t]["Recall"] if stats[t]["Recall"] > max_re else max_re
        max_f1 = stats[t]["F1-score"] if stats[t]["F1-score"] > max_f1 else max_f1

    with open("stats.json", "w") as f:
        json.dump(stats, f)

    # results
    print("full results saved into stats.json file")
    print(f"max precision with treshold {max_pr}")
    print(f"max accuracy with treshold {max_ac}")
    print(f"max recall with treshold {max_re}")
    print(f"max f1-score with treshold {max_f1}")