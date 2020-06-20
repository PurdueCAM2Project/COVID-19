import json
import argparse
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import torch

from Data_Collection_Scene_Classification.database_iterator_30kcams import database_iterator
from Data_Collection_Scene_Classification.scene_detection_30kcams import SceneDetectionClass
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

        img1 = cv2.resize(img1, (224, 224), interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, (224, 224), interpolation = cv2.INTER_AREA)
        img3 = cv2.resize(img3, (224, 224), interpolation = cv2.INTER_AREA)
        img4 = cv2.resize(img4, (224, 224), interpolation = cv2.INTER_AREA)

        diff1 = np.sum(img1 - img2)
        diff2 = np.sum(img3 - img4)
        print(diff1, diff2)
        if diff1 == 0 and diff2 == 0:
            return True
        else:
            return False
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', help='test config file path', default='Pedestron/configs/elephant/cityperson/cascade_hrnet.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='Pedestron/models_pretrained/epoch_19.pth.stu')
    args = parser.parse_args()

    i = database_iterator()
    x = SceneDetectionClass()
    print(f"total network cameras: {i.numcams}")
    cam_list_pred = dict()
    num_rand = 1
    counter = True  # False

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    detections = dict()

    count = 0

    for foldername, image_link, time in i.get_all_images():
        detections[foldername] = dict()

        print(foldername, image_link[0:1])
        check = all_same(i, image_link)
        print(check)
        if len(image_link) > 0 and not check:
            for j in range(len(image_link)):
                img = cv2.imread(i.get_image(image_link))
                results = inference_detector(model, img)
                if isinstance(results, tuple):
                    bbox_result, segm_result = results
                else:
                    bbox_result, segm_result = results, None
                bboxes = np.vstack(bbox_result)
                detections[foldername][image_link] = bboxes

    f = open("person_detections", "w")
    f.write(json.dumps(detections))
    f.close()
