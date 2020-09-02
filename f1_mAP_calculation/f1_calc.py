from xml_json import *
import os
import json
import csv

def calc_hitmiss(gt_filepath,pred_filepath,confidence_thresh): #Assumes ground truth data is xml
    """
    Calculates the number of positive and negative detections in each image"
    Args:
        gt_filepath: The filepath of the folder containing the groundtruth bounding boxes for each image in .xml
        pred_filepath: The filepath of the folder containing the predicted bounding boxes for each image in .json
        confidence_thresh: Confidence Threshold for acceptable detection. (0.0 - 1.0)
    return:
        result: a list of dictionaries with image_id, "hitcount", "misscount" for each image.
        structure of result: [{image_id1:#######,"hitcount": ##,"misscount": ##},{image_id2:#######,"hitcount": ##,"misscount": ##},]
    """
    result = []
    hitcount = 0
    misscount = 0
    gtbox_count = []
    hitbool = False
    list_index = 0
    flag = 0
    
    for image in os.listdir(pred_filepath):
        with open(pred_filepath+image) as f:
            predict_image = json.load(f)
        xml_path = gt_filepath+image
        xml_path = xml_path.replace('.json','.xml')
        gt_image = xml_json(xml_path)
        
        for confidence, predict_detection in predict_image.items():
            if float(confidence) >= confidence_thresh:
                if gt_image.get('object') is None: #No ground truth box
                    break
                elif type(gt_image['object']) is dict: #'object' value is a dict if there is only one ground truth box
                    gt_detection = list(gt_image['object']['bndbox'].values())
                    gt_detection = [float(s) for s in gt_detection]
                    iou = calc_iou(gt_detection,predict_detection)
                    if iou > 0.5:
                        hitcount += 1
                    else:
                        misscount += 1
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
                            gt_image['object'].pop(list_index) #Ground truth box is removed after a match is found
                            break
                        list_index += 1
                    list_index = 0
                    if hitbool == False:
                        misscount += 1
                    else:
                        hitbool = False
                    
        result.append({"image_id":image,"hitcount":hitcount,"misscount":misscount,"gtboxcount":gtbox_count})
        hitcount = 0
        misscount = 0
        gtbox_count = 0
        flag = 0
            
    return result 
            
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


def precision_recall(hitmisscount):
    """
    Calculates precision and recall for each image.
    Calculates the f1 score of the image set using average precision and average recall of the images.

    Args:
        hitmisscount: a list of dictionaries with image_id, "hitcount", "misscount" for each image. Return value of calc_hitmiss().
    
    return:
        f1: f1 score, the weighted average of precison and recall. (0.0 - 1.0)
    """
    ave_recall = 0
    ave_precision = 0
    count = 0
    for image in hitmisscount:
        if image["hitcount"] + image["misscount"] != 0: #zero divide prevention
            precision = image["hitcount"] / (image["hitcount"] + image["misscount"])            
            image["precision"] = precision
            
        if image["gtboxcount"] != 0:
            recall = image["hitcount"] / image["gtboxcount"]
            image["recall"] = recall
        
        if image.get("precision"):
            ave_precision += image["precision"]
            ave_recall += image["recall"]
            count += 1
    ave_precision = ave_precision / count
    ave_recall = ave_recall / count
    f1 = 2 * (ave_precision * ave_recall) / (ave_precision + ave_recall)

    return f1

#Example f1 calculation
final_f1 = precision_recall(calc_hitmiss('final_cars/final_xml/','final_cars/final_car_json_iou0.3_conf0.3/',0))
print("f1:",final_f1)

