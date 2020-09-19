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
    parser.add_argument('--start_index', required=False, type=int)
    parser.add_argument('--end_index', required=False, type=int)
    parser.add_argument('--cam_list_file', required=False, type=str)
    args = parser.parse_args()
    
    if args.cam_list_file:
	    people_6k = open('keys_people.txt', 'r')
	    lines = people_6k.read().split('\n')
	    todolist = lines
	
	elif path:
	    list_cams = os.listdir(path)
	    list_cams = [k + '/' for k in list_cams]
	    todolist = list_cams

	if args.start_index and args.end_index:
	    start_index = args.start_index
	    end_index = args.end_index
	    todolist = todolist[start_index:end_index]

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    path = args.path

    count = 0

    detections = dict()
    day_night = dict()

	    
    for cam in todolist:
        #if cam not in done:
        count+=1
        print(count)
        detections[cam] = dict()
        day_night[cam] = dict()

        for image in os.listdir(path + cam):
            month = int(image[30:32])
            day = int(image[33:35])
            print('month', month)
            print('day', day)
            if not ((month>=6 and day>=16) or (month>=7)):
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

        f = open("person_detections_MISSING_image_" + str(start_index) + "_" + str(end_index), "w+")
        f.write(json.dumps(detections))
        f.close()

        f = open("day_night_MISSING_image_" + str(start_index) + "_" + str(end_index), "w+")
        f.write(json.dumps(day_night))
        f.close()
        
