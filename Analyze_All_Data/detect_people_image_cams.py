#!/usr/bin/env python

import json
import os
import argparse
import numpy as np
import cv2
import torch
import glob
from PIL import Image


from Tools.database_iterator_30kcams import database_iterator
from Tools.scene_detection_30kcams import SceneDetectionClass
from Pedestron.mmdet.apis import init_detector, inference_detector

def all_same(i, image_link):
    if len(image_link) >= 4:
        img1 = image_link[0]
        img2 = image_link[len(image_link)//2]

        img3 = image_link[len(image_link)//4]
        img4 = image_link[len(image_link)*3//4]

        img1 = np.array(i.get_image(img1).convert('RGB'))
        img2 = np.array(i.get_image(img2).convert('RGB'))

        img3 = np.array(i.get_image(img3).convert('RGB'))
        img4 = np.array(i.get_image(img4).convert('RGB'))

        img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_AREA)
        img3 = cv2.resize(img3, (224, 224), interpolation=cv2.INTER_AREA)
        img4 = cv2.resize(img4, (224, 224), interpolation=cv2.INTER_AREA)

        diff1 = np.sum(img1 - img2)
        diff2 = np.sum(img3 - img4)
        print(diff1, diff2)
        if diff1 == 0 and diff2 == 0:
            return True
        else:
            return False
    return False

def determine_day_night(image):  # determines whether or not an image is captured during the day or night
    # 0 denotes night, 1 denotes day
    if np.mean(image) > 60:
        # this image was taken during the day
        return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run person detections on all videos')
    parser.add_argument('--config', help='test config file path', default='Pedestron/configs/elephant/cityperson/cascade_hrnet.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='Pedestron/models_pretrained/epoch_19.pth.stu')
    parser.add_argument('--path', help = 'path to videos', default='/projects/SE_HPC/covid-images/')
    parser.add_argument('--start_index', required=True, type=int)
    parser.add_argument('--end_index', required=True, type=int)
    parser.add_argument('--filename', required=True, type=str)
    parser.add_argument('--month', required=False, type=int)
    parser.add_argument('--date', required=False, type=int)
    args = parser.parse_args()
    month_int = args.month
    date_int = args.date
    filename = args.filename
    cams = open(filename, 'r')
    lines = cams.read().split('\n')
    cams = lines

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    path = args.path

    count = 0

    detections = dict()
    day_night = dict()
    
    start_index = args.start_index
    end_index = args.end_index
     
    for cam in cams[start_index:end_index]:
        #if cam not in done:
        count+=1
        print(count)
        detections[cam] = dict()
        day_night[cam] = dict()

        for image in os.listdir(path + cam):
            month = int(image[30:32])
            day = int(image[33:35])
            year = int(image[25:29])
            print('month', month)
            print('day', day)
            if not (month==1):
                continue
            print(image)
            detections[cam][image] = dict()
            day_night[cam][image] = dict()
            try:
                pil_image = Image.open(path + cam + '/' + image)
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
		# day night calculation
                if determine_day_night(img) == 0:
                    day_night[cam][image] = 'night'
                else:
                    day_night[cam][image] = 'day'
		   
                results = inference_detector(model, img)
                if isinstance(results, tuple):
                    bbox_result, segm_result = results
                else:
                    bbox_result, segm_result = results, None
                bboxes = np.vstack(bbox_result)
                bboxes = bboxes.tolist()
                bbox_dict = dict()
                for each in bboxes:
                    bbox_dict[each[4]] = each[0:4]
                detections[cam][image] = bbox_dict
            except:
                continue
        f = open('done.txt', 'a')
        f.write(cam + '/n')
        f.close()

        f = open("person_detections_image_" + "jan2021_" + str(filename) + "_" + str(start_index) + "_" + str(end_index), "w+")
        f.write(json.dumps(detections))
        f.close()

        
