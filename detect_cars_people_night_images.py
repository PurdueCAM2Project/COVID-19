import json
import argparse
import numpy as np
import cv2
import torch
import sys
import re
import os
from pathos.multiprocessing import ProcessingPool as Pool

# add the path ../ to import functions from the Pedestron module
sys.path.append("../")
sys.path.append("./")

from Tools.database_iterator_30kcams import database_iterator
from Tools.scene_detection_30kcams import SceneDetectionClass
from Pedestron.mmdet.apis import init_detector, inference_detector
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
from yolov3.detect import Vehicle_Detector


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

def main(person_model, vehicle_detector, subset_all_images, process_num):

    i = database_iterator()
    person_detections = dict()
    day_night = dict()
    vehicle_detections = dict()

    vehicle_filename = os.path.join(args.save_path, "vehicle_detections_" + process_num + ".json")
    person_filename = os.path.join(args.save_path, "person_detections_" + process_num + ".json")
    day_night_filename = os.path.join(args.save_path, "day_night_" + process_num + ".json")


    for foldername, image_link, time in i.get_subset_images(subset_all_images):

        person_detections[foldername] = dict()
        vehicle_detections[foldername] = dict()

        check = all_same(i, image_link)

        if len(image_link) > 0 and not check:
            for j in range(len(image_link)):
                pil_image = (i.get_image(image_link[j]))
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


                # day night calculation
                if determine_day_night(img) == 0:
                    day_night[foldername][image_link[j]] = 'night'
                else:
                    day_night[foldername][image_link[j]] = 'day'


                # person detection
                results = inference_detector(person_model, img)
                if isinstance(results, tuple):
                    bbox_result, segm_result = results
                else:
                    bbox_result, segm_result = results, None
                bboxes = np.vstack(bbox_result)
                bboxes = bboxes.tolist()
                bbox_dict = dict()
                for each in bboxes:
                    bbox_dict[each[4]] = each[0:4]

                person_detections[foldername][image_link[j]] = bbox_dict


                # vehicle detection
                results = vehicle_detector.detect(img, view_img=False)
                vehicle_detections[foldername][image_link] = results
                if j % 20 == 19:
                    print(f"{j + 1} done out of {len(image_link)} images")


            # write to the file at the end of every camera instead of when the entire process is complete
            # Helps if it gets disconnected in between

            f = open(vehicle_filename, "w+")
            f.write(json.dumps(vehicle_detections))
            f.close()

            f = open(person_filename, "w+")
            f.write(json.dumps(person_detections))
            f.close()

            f = open(day_night_filename, "w+")
            f.write(json.dumps(day_night))
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main script to run car, people, day night detection')
    parser.add_argument('--config', help='test config file path', default='Pedestron/configs/elephant/cityperson/cascade_hrnet.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='/local/a/cam2/data/covid19/models_pretrained/epoch_19.pth.stu')
    parser.add_argument('--cfg', type=str,
                        default='yolov3/cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str,
                        default='yolov3/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str,
                        default='yolov3/weights/yolov3-spp-ultralytics.pt', help='weights path')
    # input file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true',
                        help='half precision FP16 inference')
    parser.add_argument('--device', default='0',
                        help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--save-path', default='results',
                        help='directory to save results')
    parser.add_argument('--num_workers', type=int, default=6)

    args = parser.parse_args()

    directory_exists = os.path.isdir(args.save_path)
    if not directory_exists:
        os.mkdir(args.save_path)

    i = database_iterator()
    x = SceneDetectionClass()
    cam_list_pred = dict()
    num_rand = 1
    counter = True  # False


    torch.multiprocessing.set_start_method('spawn')

    worker_count = args.num_workers
    p = re.compile("\"(.*?)\"")
    all_images = [p.search(str(n)).group(0).strip("\"/") for n in i.folders]
    print(all_images[0])
    print(all_images)
    print(len(all_images))
    
    how_many_cams = len(all_images)
    num_folders_per_job = int(how_many_cams / worker_count)
    person_model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

   # vehicle_detector = Vehicle_Detector(weights=args.weights, cfg=args.cfg, names=args.names, iou_thres=args.iou_thres,
                      #                  conf_thres=args.conf_thres, imgsz=args.img_size, half=args.half, device_id=args.device)
    vehicle_detector = None
    worker_pool = []
    for i in range(0, worker_count):
        print('start', i*num_folders_per_job)
        print('end',  i*num_folders_per_job + num_folders_per_job + 1 )
        if i == worker_count:
            p = Process(target=main, args=(person_model, vehicle_detector, all_images[i * num_folders_per_job:], i))
        else:
            p = Process(target=main, args=(person_model, vehicle_detector, all_images[i * num_folders_per_job: i * num_folders_per_job + num_folders_per_job + 1], i))
        p.start()
        worker_pool.append(p)

    for p in worker_pool:
        p.join()


