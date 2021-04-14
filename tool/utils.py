import sys
import os
import time
import math
import cv2
from typing import Any
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from typing import List

from tqdm import tqdm


def rect_polygon(x, y, width, height, angle):
    """Return a shapely Polygon describing the rectangle with centre at
    (x, y) and the given width and height, rotated by angle quarter-turns.
    code from: https://codereview.stackexchange.com/questions/204017/intersection-over-union-for-rotated-rectangles
    """
    w = width / 2
    h = height / 2
    p = Polygon([(-w, -h), (w, -h), (w, h), (-w, h)])
    return translate(rotate(p, angle, use_radians=True), x, y)


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.0)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):

    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def my_IoU(box1: torch.Tensor, other: torch.Tensor, iou_type: str = "IoU") -> torch.Tensor:
    """Compute IoU between box1 and all the boxes in other.
    Type of IoU to run is specified with the type parameter.

    Args:
        box1 (torch.Tensor): First argument for IoU
        other (torch.Tensor): All boxes on which the IoU must be computed
        iou_type (str): type of IoU to run. Implemented: ["IoU", "gIoU", "rIoU", "rgIoU"]. Default: "IoU"

    Returns:
        torch.Tensor: list of rgIoU scores
    """

    if iou_type in ["IoU", "gIoU"]:
        # (x1,y1)----
        #   |        |
        #   |        |
        #   ------(x2,y2)

        box1_ = box1[:, :4]
        other_ = other[:, :4]

        # get coords for box1_
        box1_[:, 0] = box1[:, 0] - box1[:, 2] / 2
        box1_[:, 1] = box1[:, 0] + box1[:, 3] / 2
        box1_[:, 2] = box1[:, 1] + box1[:, 2] / 2
        box1_[:, 3] = box1[:, 1] - box1[:, 3] / 2

        # get coords for all others_
        other_[:, 0] = other[:, 0] - other[:, 2] / 2
        other_[:, 1] = other[:, 0] + other[:, 3] / 2
        other_[:, 2] = other[:, 1] + other[:, 2] / 2
        other_[:, 3] = other[:, 1] - other[:, 3] / 2

        # intersection coords
        inter_rect_x1 = torch.max(box1_[:, 0], other_[:, 0])
        inter_rect_y1 = torch.min(box1_[:, 1], other_[:, 1])
        inter_rect_x2 = torch.min(box1_[:, 2], other_[:, 2])
        inter_rect_y2 = torch.max(box1_[:, 3], other_[:, 3])

        # compute areas
        box1_area = (box1_[:, 2] - box1_[:, 0] + 1) * (box1_[:, 1] - box1_[:, 3] + 1)
        other_area = (other_[:, 2] - other_[:, 0] + 1) * (other_[:, 1] - other_[:, 3] + 1)
        inter_area = (
            torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0)
            * torch.clamp(inter_rect_y1 - inter_rect_y2 + 1, min=0)
            + 1e-16
        )
        union_area = box1_area + other_area - inter_area

        iou = inter_area / union_area
        if iou_type == "IoU":
            del box1_, other_
            return iou

        elif iou_type == "gIoU":
            # compute enclosed area
            enclos_rect_x1 = torch.min(box1_[:, 0], other_[:, 0])
            enclos_rect_y1 = torch.max(box1_[:, 1], other_[:, 1])
            enclos_rect_x2 = torch.max(box1_[:, 2], other_[:, 2])
            enclos_rect_y2 = torch.min(box1_[:, 3], other_[:, 3])
            area_c = (enclos_rect_x2 - enclos_rect_x1 + 1) * (enclos_rect_y1 - enclos_rect_y2 + 1) + 1e-16

            del box1_, other_
            return iou - (area_c - union_area) / area_c

    elif iou_type in ["rIoU", "rgIoU"]:
        box1_ = box1[:, :5]
        other_ = other[:, :5]

        # angle defined as torch.atan2(x, y) -> torch.atan2(cos,sin)
        box1_[:, 4] = torch.atan2(box1[:, 4], box1[:, 5])
        other_[:, 4] = torch.atan2(other[:, 4], other[:, 5])

        # transform boxes into polygons
        box1_ = [rect_polygon(*r) for r in box1_]
        other_ = [rect_polygon(*r) for r in other_]

        # compute areas
        box1_area = torch.Tensor([p.area for p in box1_])
        other_area = torch.Tensor([p.area for p in other_])
        inter_area = torch.Tensor([p.intersection(box1_[0]).area for p in other_]) + 1e-16
        union_area = box1_area + other_area - inter_area

        iou = inter_area / union_area
        if iou_type == "rIoU":
            return iou

        elif iou_type == "rgIoU":

            # max and min of all xy coords
            other_coords = torch.Tensor([p.exterior.coords.xy for p in other_])
            box1_coords = torch.Tensor([p.exterior.coords.xy for p in box1_])
            other_max_xy, _ = other_coords.max(2)
            other_min_xy, _ = other_coords.min(2)
            box1_max_xy, _ = box1_coords.max(2)
            box1_min_xy, _ = box1_coords.min(2)

            # compute enclosed area
            enclos_rect_x1 = torch.min(box1_min_xy[:, 0], other_min_xy[:, 0])
            enclos_rect_y1 = torch.max(box1_max_xy[:, 1], other_max_xy[:, 1])
            enclos_rect_x2 = torch.max(box1_max_xy[:, 0], other_max_xy[:, 0])
            enclos_rect_y2 = torch.min(box1_min_xy[:, 1], other_min_xy[:, 1])
            area_c = (enclos_rect_x2 - enclos_rect_x1 + 1) * (enclos_rect_y1 - enclos_rect_y2 + 1) + 1e-16

            return iou - (area_c - union_area) / area_c

    else:
        print("IoU metric not recognized")
        exit(1)


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2

    img = np.copy(img)
    colors = np.array(
        [[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        dtype=np.float32,
    )

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print("%s: %f" % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(
                img,
                class_names[cls_id],
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                rgb,
                1,
            )
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def post_processing(img, conf_thresh, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != "ndarray":
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [
                            ll_box_array[k, 0],
                            ll_box_array[k, 1],
                            ll_box_array[k, 2],
                            ll_box_array[k, 3],
                            ll_max_conf[k],
                            ll_max_conf[k],
                            ll_max_id[k],
                        ]
                    )

        bboxes_batch.append(bboxes)

    t3 = time.time()

    print("-----------------------------------")
    print("       max and argmax : %f" % (t2 - t1))
    print("                  nms : %f" % (t3 - t2))
    print("Post processing total : %f" % (t3 - t1))
    print("-----------------------------------")

    return bboxes_batch


# class to draw rotating triangles
class RotatingRectangle(Rectangle):
    def __init__(self, xy, width, height, rel_point_of_rot, **kwargs):
        """xy = center of the triangle"""
        super().__init__(xy, width, height, **kwargs)
        self.rel_point_of_rot = rel_point_of_rot
        self.xy_center = self.get_xy()
        self.set_angle(self.angle)

    def _apply_rotation(self):
        angle_rad = self.angle * np.pi / 180
        m_trans = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
        shift = -m_trans @ self.rel_point_of_rot
        self.set_xy(self.xy_center + shift)

    def set_angle(self, angle):
        self.angle = angle
        self._apply_rotation()

    def set_rel_point_of_rot(self, rel_point_of_rot):
        self.rel_point_of_rot = rel_point_of_rot
        self._apply_rotation()

    def set_xy_center(self, xy):
        self.xy_center = xy
        self._apply_rotation()


def preprocess_KITTI_input(img):
    # Kitti params
    fov = 82
    base_width = 864
    base_height = 135
    height = 136
    width = 200
    channels = 3
    canvas = np.zeros(shape=(height, width, channels), dtype=np.float)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
    img /= 255.0

    # rescale image to input size and position it to canvas center
    new_w = int(base_width * (fov / 360.0))
    new_w = new_w if new_w % 2 == 0 else new_w - 1
    new_h = int((img.shape[0] / img.shape[1]) * new_w)
    new_h = new_h if new_h % 2 == 0 else new_h - 1
    img = cv2.resize(img, (new_w, new_h))

    # define padding borders
    tb_border = (canvas.shape[0] - new_h) // 2  # top/bottom border
    lr_border = (canvas.shape[1] - new_w) // 2  # left/right border

    # fit image into canvas
    canvas[tb_border : canvas.shape[0] - tb_border, lr_border : canvas.shape[1] - lr_border, :] = img

    return canvas


def draw_bboxes_BEV(batch_size: np.ndarray, preds: torch.Tensor, fovx: float = 25.0 * 3.33) -> None:
    """draw predictions in BEV to BEV map

    Args:
        batch_size (np.ndarray): batch size
        preds (torch.Tensor): predictions of the network
        fovx (float, optional): number of cells on x axis * degree per cell. Defaults to 25.0*3.33.
    """
    plane_BEV = plt.imread("./extra_utils/BEV_plane.png")
    origin = (plane_BEV.shape[0] // 2, plane_BEV.shape[1] // 2)

    save_dir = os.path.join(os.path.abspath("."), "predictions")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(batch_size):
        mask = preds[:, 0] == float(i)
        pred = preds[mask][:, 1:]
        patches = []
        points = []

        for j, box in enumerate(pred):
            x = (box[1] * math.sin(math.radians(box[0]))) + origin[0]
            z = (box[1] * math.cos(math.radians(box[0]))) + origin[1]
            patches.append(
                RotatingRectangle(
                    (x, z),
                    width=box[3],
                    height=box[2],
                    rel_point_of_rot=(box[3] / 2, box[2] / 2),
                    angle=-math.degrees(math.atan2(box[4], box[5])),
                    color="red",
                    fill=False,
                )
            )  # minus degree because of right hand rule

            verts = patches[-1].get_verts()
            points.append(
                [
                    ((verts[2][0] - verts[1][0]) / 2) + verts[1][0],
                    ((verts[2][1] - verts[1][1]) / 2) + verts[1][1],
                ]
            )

        # draw and save BEV
        fig, ax = plt.subplots(1)
        ax.set_aspect("equal")
        ax.imshow(plane_BEV, origin="lower")
        ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], color="blue", s=2)
        for rect in patches:
            ax.add_patch(rect)
        plt.savefig(os.path.join(save_dir, str(i) + ".png"))


def nms_BEV(
    prediction: torch.Tensor,
    obj_thresh: float = 0.5,
    nms_thresh: float = 0.5,
    verbose: bool = True,
    iou_type="IoU",
) -> Any:
    """Aplly nms with rotation generalized IoU (rgIoU)

    Args:
        prediction (torch.Tensor): output of yolov4_BEV
        obj_thresh (float): confidence trashold for detection. Default = 0.3
        nms_thresh (float): nms treshold for rgIoU. Default = 0.7
        verbose (bool): verbose if True. Default = False
        iou_type (str): type of iou to use during NMS. Default = 'IoU'

    Returns:
        Any : filtered bboxes in BEV space for each image in batch.
              Format= [num_bboxes_batch * [batch_id, x,y,w,h,sin,cos,obj,conf,class]]
              Returns None if no detections have been made
    """
    if verbose:
        print("-----------------------------------")
        print(
            f"Computing NMS with nms_threshold = {nms_thresh} and obj_thesh = {obj_thresh} and IoU = '{iou_type}'"
        )
        print("WARNING: This can take a while with rIoU or rgIoU")
        print("\t Call this method with 'verbose' = False to avoid this message")
        print("-----------------------------------")

    # filter for obj_treshold
    obj_mask = (prediction[..., 6] > obj_thresh).float().unsqueeze(2)
    prediction *= obj_mask

    # loop the batch samples
    output_final = None
    for batch_id, sample_pred in tqdm(enumerate(prediction[:]), desc="Batch"):

        # instead of using all the classes, I substitute with max_conf and max_conf_idx
        if len(sample_pred[0]) > 7:
            max_conf, max_conf_idx = torch.max(sample_pred[:, 7:], 1)
        else:
            max_conf, max_conf_idx = torch.max(sample_pred[:, 6].view(-1, 1), 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_idx = max_conf_idx.float().unsqueeze(1)
        i = 7 if len(sample_pred[0]) > 7 else 6
        sample_pred = torch.cat((sample_pred[:, :i], max_conf, max_conf_idx), dim=1)

        # remove non zeros tuples over obj_thresholding
        non_zero_mask = torch.nonzero(sample_pred[:, 6])
        try:  # could give an error if image has no detections
            sample_pred = sample_pred[non_zero_mask.squeeze(1)]
        except:
            continue

        # loop over the classes
        cls_pred = torch.unique(sample_pred[:, -1])
        for cls_id in cls_pred:

            # remove bboxes that did not detect cls_id
            sample_pred_cls = sample_pred * (sample_pred[:, -1] == cls_id).float().unsqueeze(1)
            non_zero_mask = torch.nonzero(sample_pred_cls[:, 0]).squeeze()
            sample_pred_cls = sample_pred_cls[non_zero_mask]

            # order bboxes by confidence score for NMS
            conf_sort_idx = torch.argsort(sample_pred_cls[:, -3], descending=True)
            sample_pred_cls = sample_pred_cls[conf_sort_idx]

            # NMS
            for i in range(sample_pred_cls.size(0)):
                try:
                    iou_score = my_IoU(
                        sample_pred_cls[i].unsqueeze(0),
                        sample_pred_cls[i + 1 :],
                        iou_type=iou_type,
                    )
                # exceptions when we try to call the function on an index
                # that has been deleted in a previous step
                except ValueError:
                    break
                except IndexError:
                    break

                # remove tuples with less than NMS threshold of iou_score
                nms_mask = (iou_score < nms_thresh).float().unsqueeze(1)
                sample_pred_cls[i + 1 :] = sample_pred_cls[i + 1 :] * nms_mask
                nms_mask_id = torch.nonzero(sample_pred_cls[:, 0]).squeeze()
                sample_pred_cls = sample_pred_cls[nms_mask_id]

            # concatenate to output_final
            batch_id_tensor = torch.Tensor([batch_id]).repeat(sample_pred_cls.size(0)).unsqueeze(1)
            tmp_out = torch.cat((batch_id_tensor, sample_pred_cls), dim=1)

            if output_final is None:
                output_final = tmp_out
            else:
                output_final = torch.cat((output_final, tmp_out), dim=0)

    return output_final