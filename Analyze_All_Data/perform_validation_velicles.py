sys.path.append("./")
sys.path.append("../")

from yolov3.detect2 import Vehicle_Detector
import os
import sys
import json
from imageio import imread
from shutil import rmtree
import cv2
import time

image_path = 'Validation_Set/images'
result_path = 'results_vehicles'
subdirectory1 = 'already_resized_and_labeled_images_cars'
subdirectory2 = 'gt_used_images'
subdirectory3 = 'img'
img_subdirectory1 = os.path.join(image_path, subdirectory1)
img_subdirectory2 = os.path.join(image_path, subdirectory2)
img_subdirectory3 = os.path.join(image_path, subdirectory3)
res_subdirectory1 = os.path.join(result_path, subdirectory1)
res_subdirectory2 = os.path.join(result_path, subdirectory2)
res_subdirectory3 = os.path.join(result_path, subdirectory3)


if not os.path.isdir(result_path):
    os.mkdir(result_path)

if not os.path.isdir(res_subdirectory1):
    os.mkdir(res_subdirectory1)

if not os.path.isdir(res_subdirectory2):
    os.mkdir(res_subdirectory2)

if not os.path.isdir(res_subdirectory3):
    os.mkdir(res_subdirectory3)

images_names1 = os.listdir(img_subdirectory1)
images_names2 = os.listdir(img_subdirectory2)
images_names3 = os.listdir(img_subdirectory3)


imnames1 = []
imnames2 = []
imnames3 = []
images_1 = []
images_2 = []
images_3 = []


weight_cfg_pairs = [('yolov3/weights/yolov3-spp-ultralytics.pt', 'yolov3/cfg/yolov3-spp.cfg'),
                    ('yolov3/weights/yolov3-spp.pt', 'yolov3/cfg/yolov3-spp.cfg'),
                    ('yolov3/weights/yolov3.pt', 'yolov3/cfg/yolov3.cfg'),
                    ('yolov3/weights/yolov4.pt', 'yolov3/cfg/yolov4.cfg')]
iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.7]
confidence_thresholds = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
img_sizes = [1536, 1024, 512]
img_sizes = img_sizes[-1:]

for name in images_names1:
    name_ = name[:-4]
    impath = os.path.join(img_subdirectory1, name)
    image = imread(impath)
    image = cv2.imread(impath)
    image = imread(impath)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    imnames1.append(name_)
    images_1.append(image)

for name in images_names2:
    name_ = name[:-4]
    impath = os.path.join(img_subdirectory2, name)
    image = cv2.imread(impath)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    imnames2.append(name_)
    images_2.append(image)

for name in images_names3:
    name_ = name[:-4]
    impath = os.path.join(img_subdirectory3, name)
    image = cv2.imread(impath)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    imnames3.append(name_)
    images_3.append(image)

total_runs = len(weight_cfg_pairs)*len(iou_thresholds) * \
    len(confidence_thresholds)*len(img_sizes)
count = 1
for pair in weight_cfg_pairs:
    for iou in iou_thresholds:
        for conf_thresh in confidence_thresholds:
            for imgsz in img_sizes:
                print(f"Model:\t{pair[0]}")
                print(f"iou threshold:\t{iou}")
                print(f"confidence threshold:\t{conf_thresh}")
                if count % 10 == 1:
                    start = time.time()
                weight = pair[0]
                cfg = pair[1]
                output_local_dir = weight[15:]+'_iou_'+str(
                    iou)+'_conf_thresh_'+str(conf_thresh)+'_img_size_'+str(imgsz)
                output_path1 = os.path.join(
                    res_subdirectory1, output_local_dir)
                output_path2 = os.path.join(
                    res_subdirectory2, output_local_dir)
                output_path3 = os.path.join(
                    res_subdirectory3, output_local_dir)
                if not os.path.isdir(output_path1):
                    os.mkdir(output_path1)
                else:
                    rmtree(output_path1)
                    os.mkdir(output_path1)
                if not os.path.isdir(output_path2):
                    os.mkdir(output_path2)
                else:
                    rmtree(output_path2)
                    os.mkdir(output_path2)
                if not os.path.isdir(output_path3):
                    os.mkdir(output_path3)
                else:
                    rmtree(output_path3)
                    os.mkdir(output_path3)
                detector = Vehicle_Detector(
                    weights=weight, cfg=cfg, iou_thres=iou, conf_thres=conf_thresh, imgsz=imgsz)

                for file_num in range(len(imnames1)):
                    filename = imnames1[file_num]
                    file_save_path = os.path.join(
                        output_path1, filename+'.json')
                    f = open(file_save_path, 'w+')
                    image = images_1[file_num]
                    results = detector.detect(image)
                    f.write(json.dumps(results))

                for file_num in range(len(imnames2)):
                    filename = imnames2[file_num]
                    file_save_path = os.path.join(
                        output_path2, filename+'.json')
                    f = open(file_save_path, 'w+')
                    image = images_2[file_num]
                    results = detector.detect(image)
                    f.write(json.dumps(results))

                for file_num in range(len(imnames3)):
                    filename = imnames3[file_num]
                    file_save_path = os.path.join(
                        output_path3, filename+'.json')
                    f = open(file_save_path, 'w+')
                    image = images_3[file_num]
                    results = detector.detect(image)
                    f.write(json.dumps(results))

                if count % 10 == 1:
                    time_taken = time.time()-start
                    percentage_complete = (count/total_runs)*100
                    print(f"{percentage_complete}% done")
                    ETA = time_taken*(total_runs-count)
                    print(f"ETA:\t{ETA}")
                print(f"{count} done\t{total_runs-count} left")
                count = count+1
