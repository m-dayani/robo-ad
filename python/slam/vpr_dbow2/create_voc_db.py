"""
    Create small_db.yml.gz and cmd_table.txt files used for VPR Auto-Driving
"""

import argparse
import os
import sys

import numpy as np
import cv2
import imageproc  # The compiled C++ module

sys.path.append('../../')
sys.path.append('../../data_loader')
from data_loader.image_loader import ImageLoader
from data_loader.tt_loader import TabularTextLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
        Create an ORB vocabulary and DBoW2 dataset from images
    ''')
    parser.add_argument('images_path', help='images path')
    parser.add_argument('--save_path', help='save path (default: parent directory)', default='')
    args = parser.parse_args()

    images_path = args.images_path
    ds_root = os.path.split(images_path)[0]
    cmd_path = os.path.join(ds_root, 'sensors', 'cmd0.txt')

    save_path = args.save_path
    if len(save_path) <= 0:
        save_path = os.path.join(ds_root, 'dbow')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dbow_images_path = os.path.join(ds_root, 'dbow-images')
    if not os.path.exists(dbow_images_path):
        os.mkdir(dbow_images_path)

    # process images
    image_loader = ImageLoader(images_path, 'image_folder')
    images_dict = dict()
    while image_loader.is_ok():
        img_ts = image_loader.get_next()
        if img_ts is None:
            break
        ts = np.ulonglong(os.path.splitext(os.path.split(img_ts.path)[1])[0])
        images_dict[ts] = img_ts.path   # img_ts.frame

    # process ctrl commands
    cmd_loader = TabularTextLoader(cmd_path)
    cmd_table = dict()
    for cmd in cmd_loader.data:
        ts = np.ulonglong(cmd[0])
        ctrl = np.float32(cmd[1:])
        cmd_table[ts] = ctrl
    ts_cmd = np.array(sorted(cmd_table.keys()))
    cmd_loader.close()

    images_keys = sorted(images_dict.keys())
    n_images = len(images_keys)
    keys_diff = np.array(images_keys[1:]) - np.array(images_keys[:-1])
    avg_ts = sum(keys_diff) / len(keys_diff) * 1e-6
    print('Average TS: %.2f ms' % avg_ts)

    with open(os.path.join(save_path, 'cmd_table.txt'), 'w') as ts_table:
        ts_table.write('# id, trot, steer \r\n')
        for i, ts in enumerate(images_keys):
            img_file = images_dict[ts]
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img_dbow = os.path.join(dbow_images_path, 'image' + str(i) + '.png')
            cv2.imwrite(img_dbow, img)

            cmd_idx = np.argmin(abs(ts_cmd - ts))
            state = cmd_table[ts_cmd[cmd_idx]]
            ts_table.write(str(i) + ',' + ','.join(map(str, state[:2])) + '\r\n')

    # Create an instance of the C++ class
    processor = imageproc.ImageProcessor("", "")
    processor.create_voc_db(dbow_images_path, save_path, n_images)

