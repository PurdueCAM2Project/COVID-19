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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config', help='test config file path', default='Pedestron/configs/elephant/cityperson/cascade_hrnet.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='/local/a/cam2/data/covid19/models_pretrained/epoch_19.pth.stu')
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
    path = '/local/a/cam2/data/covid19/video_data/'
    count = 0
    list_cams = ['h092zALqYg', '0369289ba3', '113644aeaa', 'Sm7vwNhHoV', 'h5SGg1wbzT', 'U7REmkvwZs', '1yY7h9xkXt', '4mKEIb96LV', 'OVZjQQIIYf']

    list_cams = [k + '/' for k in list_cams]

    for cam in list_cams:
        detections[cam] = dict()
        for date in os.listdir(path + cam):
            print(date)
            detections[cam][date] = dict()
            for image in os.listdir(path + cam + date):
                detections[cam][date][image] = dict()
                pil_image = Image.open(path + cam + date + '/' + image)
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
                
                detections[cam][date][image] = bbox_dict

        f = open("person_detections_video" + cam.strip('/') , "w+")
        f.write(json.dumps(detections))
        f.close()
