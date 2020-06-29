import json
import argparse
import numpy as np
import cv2
import torch
import sys

# add the path ../ to import functions from the Pedestron module
sys.path.append("../")
sys.path.append("./")

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

def determine_day_night(image):  # determines whether or not an image is captured during the day or night
    # 0 denotes night, 1 denotes day
    if np.mean(image) > 60:
        # this image was taken during the day
        return 1
    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', help='test config file path', default='Pedestron/configs/elephant/cityperson/cascade_hrnet.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='Pedestron/pre_trained_models/epoch_19.pth.stu')
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
    day_night = dict()

    count = 0
    list_cams = ['5b194a7973569e00045d0afa', '5b194a8773569e00045d0b33', '5b194a9573569e00045d0b69', '5b19491873569e00045d0524', '5b0d3ee884d57c0004cba658', '5b0d3f7284d57c0004cba6e5', '5b19489a73569e00045d0305', '5b19788d73569e00045dc9e8', '5b19768e73569e00045db92a', '5b19726d73569e00045d9c4f', '5b0d3eb084d57c0004cba61e']
    list_cams = [k + '/' for k in list_cams]
    for foldername, image_link, time in i.get_subset_images(cam_list=list_cams):
        print(foldername)
        detections[foldername] = dict()

        check = all_same(i, image_link)
        print(check)
        if len(image_link) > 0 and not check:
            for j in range(len(image_link)):
                pil_image = (i.get_image(image_link[j]))
                #img = cv2.imread(i.get_image(image_link[0]))
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
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

                detections[foldername][image_link[j]] = bbox_dict

    f = open("person_detections", "w")
    f.write(json.dumps(detections))
    f.close()
