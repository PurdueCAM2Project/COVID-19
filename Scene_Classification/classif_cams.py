import json
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import imageio

from Tools.database_iterator_30kcams import database_iterator
from Tools.scene_detection_30kcams import SceneDetectionClass

def brightness(image): 
    img = np.array(image.convert('RGB')) 
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    # black_or_white = np.mean(img)
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
    img_arr = np.array(img)
    thresh = np.sum(img_arr)/(224 * 224)
    
    # print(f"thresh: {thresh}")
    # print(f"b/w : {black_or_white}")
    return thresh, 100#black_or_white

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

def validate_image(dbi, image, foldername):
    thresh, black_or_white = brightness(image)
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
        thresh, black_or_white = brightness(image)

        # if black_or_white < 37 or black_or_white > 240:
        #     timeout_limit = 1
        # else:
        #     timeout_limit = 10

        timeout += 1
        if timeout > timeout_limit:
            break
    return image

def classify_image(i, x, image_link, foldername, show_image = True):
    img_nam = i.get_image(image_link)
    if (img_nam == None):
        return None, None
    img_nam = validate_image(i, img_nam, foldername)
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
    num_rand = 1
    counter = True#False

    count = 0
    try:
        for foldername, image_link, time in i.get_n_arbitrary_images(num_rand=num_rand):
            print(foldername, image_link[0:1])
            check = all_same(i, image_link)
            print(check)
            if len(image_link) > 0 and not check:
                for j in range(len(image_link)):      
                    top_pred, attributes = classify_image(i, x, image_link[j], foldername, show_image = counter)
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