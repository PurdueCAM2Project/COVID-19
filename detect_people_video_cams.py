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
    parser = argparse.ArgumentParser(description='Run person detections on all videos')
    parser.add_argument('--config', help='test config file path', default='Pedestron/configs/elephant/cityperson/cascade_hrnet.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='/local/a/cam2/data/covid19/models_pretrained/epoch_19.pth.stu')
    parser.add_argument('--path', help = 'path to videos', default='/local/a/cam2/data/covid19/video_data/')
    args = parser.parse_args()

    i = database_iterator()
    x = SceneDetectionClass()
    print(f"total network cameras: {i.numcams}")
    cam_list_pred = dict()
    num_rand = 1
    counter = True  # False

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    path = args.path

    count = 0

    list_cams = os.listdir(path)

    # list_cams = ['h092zALqYg', '0369289ba3', '113644aeaa', 'Sm7vwNhHoV', 'h5SGg1wbzT', 'U7REmkvwZs', '1yY7h9xkXt', '4mKEIb96LV', 'OVZjQQIIYf']

    list_cams = [k + '/' for k in list_cams]
    
    print(list_cams)
    detections = dict()
    day_night = dict()

    for cam in list_cams
        detections[cam] = dict()
        day_night[cam] = dict()

        for date in os.listdir(path + cam):
            print(date)
            detections[cam][date] = dict()
            day_night[cam][date] = dict()

            for image in os.listdir(path + cam + date):
                detections[cam][date][image] = dict()
                day_night[cam][date][image] = dict()

                pil_image = Image.open(path + cam + date + '/' + image)
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # day night calculation
                if determine_day_night(img) == 0:
                    day_night[cam][date][image] = 'night'
                else:
                    day_night[cam][date][image] = 'day'

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
                
                detections[cam][date][image] = bbox_dict

        f = open("person_detections_video", "w+")
        f.write(json.dumps(detections))
        f.close()

        f = open("day_night_video_detections", "w+")
        f.write(json.dumps(day_night))
        f.close()
