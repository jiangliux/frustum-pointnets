#/bin/bash
python3 train/test.py --gpu 2 --num_point 1024 --model frustum_pointnets_gcnn --model_path train/log_gcnn_complete/model.ckpt --output train/detection_results_gcnn_complete --data_path kitti/frustum_carpedcyc_val_rgb_detection.pickle --idx_path kitti/image_sets/val.txt --from_rgb_detection
train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_gcnn_complete
