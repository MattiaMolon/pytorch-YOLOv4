from tool.utils import *
import torch
import pandas as pd
from tool.darknet2pytorch import Darknet
import os
import cv2
from tqdm import tqdm

# check for cuda
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# params
input_type = "KITTI_canvas"
checkpoint_dir = "KITTI_canvas_freeze"
train_dir = "../data/KITTI/train/images"
splits_dir = "../data/KITTI/splits"
names_path = "./names/BEV.names"
iou_type = "rgIoU"
iou_thres = -0.5
conf_thres = 0.9
draw_BEV = False


if __name__ == "__main__":
    if input_type == "KITTI_basic":
        cfg_model_file = "./cfg/model/yolov4_BEV_flat_KITTI_basic.cfg"
        weights = f"./checkpoints/{checkpoint_dir}/Yolov4_BEV_flat_epoch133.pth"
    elif input_type == "KITTI_canvas":
        cfg_model_file = "./cfg/model/yolov4_BEV_flat_KITTI_canvas.cfg"
        weights = f"./checkpoints/{checkpoint_dir}/Yolov4_BEV_flat_epoch79.pth"

    # names file
    mapping = {}
    with open(names_path, "r") as f:
        for i, line in enumerate(f):
            mapping[line] = float(i)

    # model
    m = Darknet(cfg_model_file, model_type="BEV_flat", inference=True)
    m.load_state_dict(torch.load(weights, map_location=torch.device(device)))
    m.eval()

    # dataframes
    columns = ["ID", "alpha", "3D_d", "3D_w", "3D_l", "sin", "cos", "cls"]
    types = {
        "ID": str,
        "alpha": float,
        "3D_d": float,
        "3D_w": float,
        "3D_l": float,
        "sin": float,
        "cos": float,
        "cls": str,
    }
    df_gt = pd.read_csv(os.path.join(splits_dir, "val_split.csv"), dtype=types)
    df_gt = df_gt.replace({"cls": mapping})
    df_gt.astype({"cls": float})
    df_preds = pd.DataFrame(columns=["ID", "conf", "correct"])

    # generate preds
    with open(os.path.join(splits_dir, "val_split.txt"), "r") as paths:
        print("Generating predictions...")
        for i, path in tqdm(enumerate(paths), ncols=100, leave=False):

            # preprocess input
            id_img = path.strip().split("/")[-1].split(".")[0]
            input = cv2.imread(os.path.join(train_dir, id_img + ".png"))
            input = preprocess_KITTI_input(input, input_type=input_type)
            input = torch.from_numpy(input).unsqueeze(0)
            input = input.permute(0, 3, 1, 2).float().to(device)

            preds = m(input).detach()

            # draw BEV
            if draw_BEV:
                nms_preds = nms_BEV(preds, iou_type=iou_type, obj_thresh=0.8, verbose=False)
                draw_bboxes_BEV(input.shape[0], nms_preds, names=[id_img])

            # save predictions to pandas dataframe
            preds = preds.squeeze()
            mask_conf = torch.nonzero((preds[:, -1] >= conf_thres).float()).squeeze()
            preds = preds[mask_conf]
            if preds.shape[0] == 0:
                continue
            if len(preds.shape) == 1:
                preds = preds.unsqueeze(0)

            # save preds
            gts = df_gt[df_gt["ID"] == id_img]
            if gts.empty:
                for pred in preds:
                    df_preds = df_preds.append(
                        pd.Series({"ID": id_img, "conf": pred[-1].item(), "correct": 0.0}),
                        ignore_index=True,
                    )
            else:
                for pred in preds:
                    try:
                        iou_scores = my_IoU(
                            pred.unsqueeze(0), torch.from_numpy(gts[gts.columns[1:]].to_numpy()), iou_type
                        )
                    except Exception:
                        break

                    correct = (iou_scores >= iou_thres).any().float().item()
                    df_preds = df_preds.append(
                        pd.Series({"ID": id_img, "conf": pred[-1].item(), "correct": correct}),
                        ignore_index=True,
                    )

    df_preds.astype({"ID": str, "conf": float, "correct": float})
    print("Computing precision recall curve...")

    # computing precision / recall
    n_gt = len(df_gt)
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

    plt.plot(recall, precision)
    plt.show()

    # computing interpolated AP (VOC format)
    ap_voc = 0
    prev_i, prev_prec = 0, precision[0]
    for i, prec in enumerate(precision[1:]):
        if prec < prev_prec:
            ap_voc += (recall[i] - recall[prev_i]) * prev_prec
            prev_i = i
            prev_prec = prec

    print(f"AP_voc : {ap_voc}")