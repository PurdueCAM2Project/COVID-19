from xml_json import *
import os
import json
import csv
import numpy as np

def calc_hitmiss(gt_filepath,pred_filepath,confidence_thresh): 
    """
    Calculates the mean average precision of the image set using the Area Under Curve method
    Args:
        gt_filepath: The filepath of the folder containing the groundtruth bounding boxes for each image in .xml
        pred_filepath: The filepath of the folder containing the predicted bounding boxes for each image in .json
        confidence_thresh: Confidence Threshold for acceptable detection. (0.0 - 1.0)
    return:
        mean_ave_prec: Mean Average Precision of the image set
    """
    hitcount = 0
    misscount = 0
    gtbox_count = []
    hitbool = False
    list_index = 0
    flag = 0
    pred_count = 0
    hitmisslist = []
    ave_prec_list = []
    
    for image in os.listdir(pred_filepath):
        with open(pred_filepath+image) as f:
            predict_image = json.load(f)
        xml_path = gt_filepath+image
        xml_path = xml_path.replace('.json','.xml')
        gt_image = xml_json(xml_path)
        
        for confidence, predict_detection in predict_image.items():
            if float(confidence) >= confidence_thresh:
                pred_count += 1
                if gt_image.get('object') is None: #No gt box
                    break
                elif type(gt_image['object']) is dict: #'object' value is a dict if only one gt box
                    gt_detection = list(gt_image['object']['bndbox'].values())
                    gt_detection = [float(s) for s in gt_detection]
                    iou = calc_iou(gt_detection,predict_detection)
                    if iou > 0.5:
                        hitcount += 1
                        hitmisslist.append(1)
                    else:
                        misscount += 1
                        hitmisslist.append(0)
                    gtbox_count = 1
                elif type(gt_image['object']) is list:
                    if flag == 0:
                        gtbox_count = len(gt_image['object']) #Prevent gtbox_count change after boxes are removed
                        flag = 1
                    for gtbox in gt_image['object']: #'object' is a list of dict if more than one gt box
                        gt_detection = list(gtbox['bndbox'].values())
                        gt_detection = [float(s) for s in gt_detection]
                        iou = calc_iou(gt_detection,predict_detection)
                        if iou > 0.5:
                            hitcount += 1
                            hitbool = True
                            gt_image['object'].pop(list_index)
                            hitmisslist.append(1)
                            break
                        list_index += 1
                    list_index = 0
                    if hitbool == False:
                        misscount += 1
                        hitmisslist.append(0)
                    else:
                        hitbool = False
        tp = 0
        tp_fp = 0
        prec = []
        rec= []
        for i in hitmisslist:
            tp += i
            tp_fp += 1
            prec.append(tp/tp_fp)
            if gtbox_count != 0:
                rec.append(tp/gtbox_count)
            else:
                rec.append(0)
        ave_prec = 0
        for i in range(len(prec)):
            if i == 0:
                ave_prec += prec[0] * rec[0]
            else:
                ave_prec += (rec[i] - rec[i - 1]) * prec[i]
        if gtbox_count != 0:
            ave_prec_list.append(ave_prec)
        hitcount = 0
        misscount = 0
        gtbox_count = 0
        flag = 0
        hitmisslist = []
        pred_count = 0
    mean_ave_prec = np.mean(ave_prec_list)
     
    return mean_ave_prec
            
def calc_iou(gt,pred):
    """
    Calculates intersection over union for given ground truth box and predicted box.
    Args:
        gt: List of coordinates of the ground truth box [X_min,Y_min,X_max,Y_max]
        pred: List of coordinates of the predicted box [X_min,Y_min,X_max,Y_max]
    return:
        iou: Intersection Over Union Value (0.0 - 1.0)
    """
    #Calculate intersection
    xA = max(gt[0],pred[0]) #Xmin of intersection box
    yA = max(gt[1],pred[1]) #Ymin 
    xB = min(gt[2],pred[2]) #Xmax
    yB = min(gt[3],pred[3]) #Ymax

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    #Calculate union
    gtArea = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
    predArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    union = float(gtArea + predArea - interArea)

    #Calculate iou
    iou = interArea / union

    return iou

##Example mAP calculation
final_map = calc_hitmiss('final_cars/final_xml/','final_cars/final_car_json_iou0.3_conf0.3/',0)
print("mAP:",final_map)
