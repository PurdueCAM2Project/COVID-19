import argparse

import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import cv2
import torch
import glob
import json
import mmcv
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('output_dir', type=str, help='the dir for result images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    parser.add_argument('--show_results', default=False)
    parser.add_argument('--save_json', default=True)
    parser.add_argument('--output_json_dir', type=str)
    args = parser.parse_args()
    return args
"""
def run_all_videos():
    args = parse_args()
    video_directory = args.video_dir_name
    cam_id = [f.path for f in os.scandir(video_directory) if f.is_dir()]
    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    results_dict = {}
    for cam in cam_id:
        video_timestamp = [g.path for g in os.scandir(cam_id) if g.is_dir()]
        results_dict[cam] = 0

        for video in video_timestamp:
            eval_imgs = glob.glob(os.path.join(video_timestamp, '*.png'))
    
            for image in eval_imgs:
                mock_detector(model, image, None):
                    img = cv2.imread(image)
                    results = inference_detector(model, image)
                    if isinstance(results, tuple):
                        bbox_result, segm_result = results
                    else:
                        bbox_result, segm_result = results, None
                    bboxes = np.vstack(bbox_result)
"""
def call_mock_detector_json():
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.png'))

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    prog_bar = mmcv.ProgressBar(len(eval_imgs))
    for im in eval_imgs:
        mock_detector_json(model, im, output_dir)
        prog_bar.update()


def mock_detector_json(model, image_name, output_dir):
    image = cv2.imread(image_name)
    results = inference_detector(model, image)
    if isinstance(results, tuple):
        bbox_result, segm_result = results
    else:
        bbox_result, segm_result = results, None
    bboxes = np.vstack(bbox_result)
    try:
        os.mkdir(output_dir)
    except OSError:
        print('couldnt make directory')
    simple_image_name = image_name.split('/')[-1].split('.')[0]
    result_name = os.path.join(output_dir, simple_image_name + '.json')
    result_txtfile = open(result_name, 'w+')
    bbox_list = bboxes.tolist()
    bbox_dict = {}
    for each in bbox_list:
        bbox_dict[each[4]] = each[0:4]
    result_txtfile.write(json.dumps(bbox_dict))
    result_txtfile.close()
            
def mock_detector(model, image_name, output_dir):
    image = cv2.imread(image_name)
    results = inference_detector(model, image)
    
    basename = os.path.basename(image_name).split('.')[0]
    
    result_name = basename + "_result.jpg"
    result_name = os.path.join(output_dir, result_name)
    show_result(image, results, model.CLASSES, out_file=result_name)

def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

def run_detector_on_dataset():
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.png'))

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    prog_bar = mmcv.ProgressBar(len(eval_imgs))
    for im in eval_imgs:
        detections = mock_detector(model, im, output_dir)
        prog_bar.update()

if __name__ == '__main__':
    args = parse_args()
    if args.save_json:
        call_mock_detector_json()
    else:
        run_detector_on_dataset()
