import argparse
import cv2
import os
import numpy as np
import kitti.kitti_util as utils
from kitti.kitti_object import kitti_object_video, run_on_opencv_image, kitti_object, run_on_tracking_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def demo_video():
    video_path = os.path.join(ROOT_DIR, 'dataset/KITTI/2011_09_26/')

    dataset = kitti_object_video(os.path.join(video_path, '2011_09_26_drive_0001_sync/image_03/data'),
                                 os.path.join(video_path, '2011_09_26_drive_0001_sync/velodyne_points/data'),
                                 video_path, )

    result_dir = '../train/detection_results_video'
    det_dir = os.path.join(result_dir, 'data')

    for id in range(dataset.__len__()):
        img = dataset.get_image(id)
        calib = dataset.get_calibration(id)
        det_file = os.path.join(det_dir, os.path.basename(dataset.img_filenames[id]).rstrip('.png') + '.txt')
        with open(det_file, 'r') as f:
            lines = f.readlines()
            objs = [utils.Object3d(line, from_detection=True) for line in lines]

        img1, img2 = run_on_opencv_image(img, objs, calib)
        cv2.imshow("3D detection", np.hstack((img1, img2)))
        if cv2.waitKey(1000) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

def demo_single_image():
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split='training')
    image_set_file = os.path.join(ROOT_DIR, 'kitti/image_sets/val.txt')

    image_set = [int(line.rstrip('\n')) for line in open(image_set_file, 'r', encoding='utf-8').readlines()]
    result_dir = '../train/detection_results_gcnn'
    det_dir = os.path.join(result_dir, 'data')

    for id in range(len(image_set)):
        true_id = image_set[id]
        img = dataset.get_image(true_id)
        calib = dataset.get_calibration(true_id)
        det_file = os.path.join(det_dir, '%06d.txt' %true_id)
        label_file = os.path.join(dataset.label_dir, '%06d.txt' %true_id)
        with open(label_file, 'r') as f:
            lines = f.readlines()
            objs = [utils.Object3d(line, from_detection=False) for line in lines]
        img1, img2 = run_on_opencv_image(img, objs, calib)
        # cv2.imwrite(os.path.join('/data1/jiang/demo', 'detection_{}.png'.format('%06d' %true_id)), img2)
        cv2.imwrite(os.path.join('/data1/jiang/gt', 'gt_{}.png'.format('%06d' %true_id)), img2)

        # cv2.imshow("2D detection", img1)
        # cv2.imshow("3D detection", img2)
        # cv2.waitKey()

def run_all_sequence(kitti_dir='/data1/KITTI/2011_09_28', is_hypotheses=True):
    for sequence in os.listdir(kitti_dir):
        sequence = '2011_09_28_drive_0039_sync'
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(kitti_dir, sequence)
        if not os.path.isdir(sequence_dir):
            continue

        image_dir = os.path.join(sequence_dir, "image_03/data")
        lidar_dir = os.path.join(sequence_dir, "velodyne_points/data")
        if is_hypotheses:
            det_dir = os.path.join(sequence_dir, 'det/hypotheses_3D/data')
        else:
            det_dir = os.path.join(sequence_dir, 'det/detections_3D/data')

        calib_dir = kitti_dir
        dataset = kitti_object_video(image_dir, lidar_dir, calib_dir)
        for id in range(dataset.__len__()):
            print(id)
            img = dataset.get_image(id)
            calib = dataset.get_calibration(id)
            det_file = os.path.join(det_dir, os.path.basename(dataset.img_filenames[id]).rstrip('.png') + '.txt')
            with open(det_file, 'r') as f:
                lines = f.readlines()
                objs = [utils.Object3d(line, from_detection=False, from_tracking=is_hypotheses) for line in lines]

            img1, img2 = run_on_tracking_image(img, objs, calib)
            cv2.imshow("3D detection", np.vstack((img1, img2)))
            if cv2.waitKey(500) == 27:
                break  # esc to quit
            # output_prefix = os.path.join('/data1/jiang/tmp', sequence)
            # cv2.imwrite(output_prefix + '_2D_' + str(id) + '.png', img1)
            # cv2.imwrite(output_prefix + '_3D_' + str(id) + '.png', img2)

        cv2.destroyAllWindows()
        break

if __name__ == '__main__':
    # demo_single_image()
    run_all_sequence()