import json
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import imageio

import torch
from torchvision import transforms as trn

from database_iterator_30kcams import database_iterator
from scene_detection_gpu import SceneDetectionClass

def brightness(image, tf): 
    img = tf(image).cuda()
    thresh = torch.mean((img > 0.5).float() * 255)
    return thresh

def transform():
    tf = trn.Compose([
        trn.Resize((224, 224)), 
        trn.ToTensor()
    ])
    return tf

def transform_grey():
    tf = trn.Compose([
        trn.Resize((224, 224)), 
        trn.Grayscale(num_output_channels=1),
        trn.ToTensor()
    ])
    return tf

def all_same(i, image_link, tf):
    if len(image_link) >= 4:
        img1 = image_link[0]
        img2 = image_link[len(image_link)//2]

        img3 = image_link[len(image_link)//4]
        img4 = image_link[len(image_link)*3//4]

        img1 = tf(i.get_image(img1))#.cuda()
        img2 = tf(i.get_image(img2))#.cuda()
        img3 = tf(i.get_image(img3))#.cuda()
        img4 = tf(i.get_image(img4))#.cuda()

        diff1 = torch.sum(img1 - img2).item()
        diff2 = torch.sum(img3 - img4).item()

        print(diff1, diff2)
        if diff1 == 0 and diff2 == 0:
            return True
        else:
            return False
    return False

def merge_classif(current, cam, addition):
    if cam not in current.keys():
        current[cam] = {key: [value, 1] for key, value in addition.items()}
    else:
        for key in addition.keys():
            if key in current[cam]:
                current[cam][key][0] += addition[key]
                current[cam][key][1] += 1
            else:
                current[cam][key] = [addition[key], 1]
    return current

def validate_image(dbi, image, foldername, tf):
    thresh = brightness(image, tf)
    timeout = 0
    timeout_limit = 10
    number = None


    if type(dbi.random_select) != int:
        number = False
    else:
        number = True
        
    while thresh < 50 or thresh > 240:
        new_ind = np.random.randint(0, len(dbi.imgs))
        tmout2 = 0
        if number:
            while (new_ind == dbi.random_select):
                new_ind = np.random.randint(0, len(dbi.imgs))
                tmout2 += 1
                if tmout2 > 1000:
                    break
            dbi.random_select = [dbi.random_select, new_ind]
        else:
            dbi.random_select = list(dbi.random_select)
            while (new_ind in dbi.random_select):
                new_ind = np.random.randint(0, len(dbi.imgs))
                tmout2 += 1
                if tmout2 > 1000:
                    break
            dbi.random_select.append(new_ind)
        #print(dbi.random_select)
        image = dbi.imgs[new_ind]
        image = dbi.get_image(image, folder_name = foldername)
        thresh = brightness(image, tf)

        # if black_or_white < 37 or black_or_white > 240:
        #     timeout_limit = 1
        # else:
        #     timeout_limit = 10

        timeout += 1
        if timeout > timeout_limit:
            break
    return image

def classify_image(i, x, image_link, foldername, tf, show_image = True):
    img_nam = i.get_image(image_link)
    if (img_nam == None):
        return None, None
    img_nam = validate_image(i, img_nam, foldername, tf)
    image = x.set_image(img_nam)
    top_pred, attributes = x.run(supress_printing = True, supress_images = show_image)
    return top_pred, attributes

def dict_mean_sort(classif, length):
    for key in classif.keys():
        classif[key] = [float(val/length) for val in classif[key]]

    items = list(classif.items())
    items.sort(key=lambda x: x[1][1], reverse = True)
    
    ret_dict = dict()
    count = 0
    for key, value in items:
        ret_dict[key] = value
        count += 1
        # if count > 6:
        #     break
        
    # print(ret_dict)
    return ret_dict


if __name__ == "__main__":
    i = database_iterator()
    x = SceneDetectionClass()
    print(f"total network cameras: {i.numcams}")
    cam_list_pred = dict()
    num_rand = 5
    counter = True#False

    tf = transform()
    tf_g = transform_grey()

    count = 0
    try:
        for foldername, image_link, time in i.get_n_arbitrary_images(num_rand=num_rand):
            print(foldername, image_link[0:1])
            check = all_same(i, image_link, tf)
            print(check)
            if len(image_link) > 0 and not check:
                for j in range(len(image_link)):      
                    top_pred, attributes = classify_image(i, x, image_link[j], foldername, tf_g, show_image = counter)
                    cam_list_pred = merge_classif(cam_list_pred, foldername, top_pred)
                    counter = True
                counter  = True#False
                cam_list_pred[foldername] = dict_mean_sort(cam_list_pred[foldername], len(image_link))
            elif check:
                cam_list_pred[foldername] = {"dead-cam":[1, -999]}
            else:
                cam_list_pred[foldername] = {"camera_empty":[1, -999]}
            print(f"folder {foldername} : {cam_list_pred[foldername]}")
            count += 1
            print(f"cam number: {count}\n")
    except KeyboardInterrupt:
        for key in cam_list_pred.keys():
            print(f"{key}:{cam_list_pred[key]}")
        f = open("classifications", "w")
        f.write(json.dumps(str(cam_list_pred)))
        f.close()
        raise
    except:
        for key in cam_list_pred.keys():
            print(f"{key}:{cam_list_pred[key]}")
        f = open("classifications", "w")
        f.write(json.dumps(str(cam_list_pred)))
        f.close()
        raise

        f = open("classifications", "w")
        f.write(json.dumps(cam_list_pred))
        f.close()