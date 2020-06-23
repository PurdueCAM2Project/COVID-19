import json
import argparse
import numpy as np
import cv2
import torch
import sys
import matplotlib.pyplot as plt
import time
# add the path ../ to import functions from the yolov3 module
sys.path.append("../")
sys.path.append("./")

from yolov3.utils.datasets import *
from yolov3.utils.utils import *
from yolov3.detect import Vehicle_Detector
from Tools.scene_detection_30kcams import SceneDetectionClass
from Tools.database_iterator_30kcams import database_iterator

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO People Detector')
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
    args = parser.parse_args()
    args.cfg = check_file(args.cfg)  # check file
    args.names = check_file(args.names)  # check file
    print("Yolo for vehicle detection configuration:")
    print(args)
    directory_exists = os.path.isdir(args.save_path)
    if not directory_exists:
        os.mkdir(args.save_path)

    i = database_iterator()
    print(f"total network cameras: {i.numcams}")

    vehicle_detector = Vehicle_Detector(weights=args.weights, cfg=args.cfg, names=args.names, iou_thres=args.iou_thres,
                                        conf_thres=args.conf_thres, imgsz=args.img_size, half=args.half, device_id=args.device)

    detections = dict()

    count = 0
    filename = os.path.join(args.save_path, "vehicle_detections.json")

    list_cams = ['5b194a7973569e00045d0afa', '5b194a8773569e00045d0b33', '5b194a9573569e00045d0b69', '5b19491873569e00045d0524', '5b0d3ee884d57c0004cba658', '5b0d3f7284d57c0004cba6e5', '5b19489a73569e00045d0305', '5b19788d73569e00045dc9e8', '5b19768e73569e00045db92a', '5b19726d73569e00045d9c4f', '5b0d3eb084d57c0004cba61e']
    list_cams = [k + '/' for k in list_cams]
    for foldername, image_links, time in i.get_subset_images(cam_list=list_cams):
        detections[foldername] = dict()
        print(foldername, image_links[:1])
        check = all_same(i, image_links)
        print(f"number of images in this camera:\t{len(image_links)}")
        if check:
            print("All images are same for this camera. Skipping...")
        if len(image_links) > 0 and not check:
            for j in range(len(image_links)):
                image_link = image_links[j]
                img = np.array(i.get_image(image_link).convert('RGB'))
                results = vehicle_detector.detect(img, view_img=False)
                detections[foldername][image_link] = results
                if j%20==19:
                    print(f"{j+1} done out of {len(image_links)} images")

            f = open(filename, "w+")
            # write to the file at the end of every camera instead of when the entire process is complete
            # Helps if it gets disconnected in between
            print(detections)
            f.write(json.dumps(detections))
            f.close()
            count += 1
            print(f"{count} out of {i.numcams} cameras done.")
    print("Done")
