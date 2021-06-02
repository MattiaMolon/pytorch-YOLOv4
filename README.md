# Pytorch-YOLOv4_BEV

## Common
- [x] rIoU and rgIoU implementation
- [x] NMS implementation
- [x] Understand structure
- [x] Adapt to KITTI dataset
- [x] Adapt to nuScenes dataset
- [x] Validation AP

## Yolov4_BEV_grid
- [x] Build structure
- [X] Implement forward pass
- [ ] Training loop
- [ ] Loss implementation
- [ ] BEV visualization integration

## Yolov4_BEV_flat
- [x] Build structure
- [x] Implement forward pass
- [x] Training loop
- [x] Loss implementation
- [x] BEV visualization integration
- [x] Loss iou test  

## Yolov4_BEV_dist
- [x] Build structure
- [x] Implement forward pass
- [x] Training loop
- [x] Loss implementation
- [x] visualization integration

## How to use demos and scripts
run demos and scripts on the root directory by running "python <folder>/<python file name>"
Tested on:
- [x] demos/demo_yolov4_BEV_dist.py
- [x] demos/demo_yolov4_BEV_flat.py
- [ ] demos/demo_yolov4_BEV_grid.py
- [x] demos/demo_yolov4.py
- [x] scripts/predict_and_save_yolov4.py
- [ ] scripts/compute_AP_flat.py
- [ ] scripts/compute_distance_loss.py

## How to train
move from train_files to the root directory the train file of interest and then run "python train_<train name>"