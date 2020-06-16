import requests
import os
from bs4 import BeautifulSoup
import re
import numpy as np
import datetime as dt
import urllib.request
from PIL import Image
import cv2
import imageio
import numpy as np 


"""
Writers: 
    Vishnu Banna

Contact: 
    DM cam2 slack: 
        Vishnu Banna
    email: 
        vbanna@purdue.edu

requires > 8gb of memory
"""

"""
SAMPLE CODE:
-----------

i = database_iterator()
print(f"total network cameras: {i.numcams}")

for foldername, image_link, time in i.get_arbitrary_images():
    print(image_link, time)
    print(i.get_image(image_link).size)

"""
class database_video_iterator():
    """
    Purpose: 
        a set of iterators to pull images from the http://vision.cs.luc.edu/~cv/images/ network cameras data base
    """

    def __init__(self, link = 'http://vision.cs.luc.edu/~cv/videos/'):
        """
        Purpose: 
            init iterators
        
        input: 
            link: link to database

        return: 
            None
        """
        self.link = link
        self.folders = self.get_folders()
        self.numcams = len(self.folders)
        self.imgs = None
        self.dtms = None
        self.random_select = None
        return
    
    @property
    def folder_names(self):
        folders = []
        for folder in self.folders:
            folders.append(folder.text)
        return folders

    def get_folders(self):
        """
        Purpose: 
            get the cameras in the database
        
        input: 
            None

        return: 
            folders: the list of soup objects for each folder/directory urls, in order of reading 
        """
        html_text = requests.get(self.link).text
        soup = BeautifulSoup(html_text, 'html.parser')
        print(soup)
        print(len(soup))
        #Getting only the Folder directories 
        attrs = {
            'href': re.compile(r'[\w]+')
        }
        folders = soup.find_all('a', attrs = attrs)#only folder files
        return folders
    
    def get_images(self, directory):
        """
        Purpose: 
            get the images in the directory
        
        input: 
            directory: the image directory in the data base

        return: 
            files: the list of soup objects for each image url, in order of reading 
        """
        html_text = requests.get(f"{self.link}{directory}").text
        soup = BeautifulSoup(html_text, 'html.parser')
        attrs = {
            'href': re.compile('([^\s]+(\.(?i)(mp4|mov))$)')
        }
        files = soup.find_all('a', attrs=attrs)
        return files
    
    def get_datetimes(self, directory):
        """
        Purpose: 
            get the data and time of each image in directory
        
        input: 
            directory: the image directory in the data base

        return: 
            files: the list of datas and times for each image, in order of reading 
        """
        html_text = requests.get(f"{self.link}{directory}").text
        soup = BeautifulSoup(html_text, 'html.parser')
        tsearch = re.compile(r'[\d]+-[\d]+-[\d]+ [\d]+:[\d]+')
        files = soup.find_all('td', text=tsearch)
        return files

    def get_video_frame(self, image, url = True, folder_name = None):
        # name="http://vision.cs.luc.edu/~cv/videos/bcfe3110b5/2020-05-11_23_54_52.mp4"
        if image == None:
            return None
        if folder_name != None:
            image = f"{self.link}/{folder_name}/{image['href']}"
        
        if url == True:
            image = imageio.get_reader(image)
            frames = []
            for i, frame in enumerate(image):
                frames.append(frame)
                if i > 5:
                    break
            del image
        else:
            image = imageio.get_reader(image)
            frames = []
            for i, frame in enumerate(image):
                frames.append(frame)
                if i > 5:
                    break
            del image
        
    def get_all_frames(self, image, url = True, folder_name = None, filepath = "None", iter_space = 1):
        # name="http://vision.cs.luc.edu/~cv/videos/bcfe3110b5/2020-05-11_23_54_52.mp4"
        print(image)
        if image == None:
            return None
        if folder_name != None:
            image = f"{self.link}/{folder_name}/{image['href']}"
        # frames = []
        #image = imageio.get_reader(image)
        #for i, frame in enumerate(image):
        #frames = list(image)
        #del image
        cap = cv2.VideoCapture(image)
        count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                if filepath == None:
                    cv2.imwrite("frame%d.png" % count, frame)
                else:
                    cv2.imwrite(f"{filepath}/frame{count}.png", frame)
                count += iter_space
                cap.set(1, count)
            else:
                cap.release()
        return filepath

    def get_link(self, folder, image):
        return f"{self.link}{folder}/{image}"

    def iterate_folder(self, folder_name, starting = 0):
        self.imgs = self.get_images(folder_name)
        self.dtms = self.get_datetimes(folder_name)

        if len(self.imgs) <= 0:
            yield None, None
        else:
            for i in range(starting, len(self.imgs)):
                yield self.get_link(folder_name, self.imgs[i].text), self.dtms[i]
    


# from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def get_hw(image, width):
    sp = image.shape[:2] 
    height= int(sp[0]/sp[1] * width)
    return height


def save_movpeg(folder, image, number):
    if not os.path.isdir(f"videos/{folder}"):
        os.mkdir(f"videos/{folder}")


if __name__ == "__main__":
    k = database_video_iterator()
    folder_names = k.folder_names
    clear = True
    print(folder_names)

    if os.path.isdir(f"videos/"):
        if clear:
            os.system(f"rm -r videos/*")
            os.system(f"rmdir videos/*")
    else:
        os.mkdir("videos")
        
    for folder_name in folder_names:
        os.mkdir(f"videos/{folder_name}")
        for cam, time in k.iterate_folder(folder_name = folder_name):
            try:
                if cam != None:
                    fname = cam.split("/")[-1].split(".")[0]
                    path = f"videos/{folder_name}/{fname}"
                    os.mkdir(path)
                    image = k.get_all_frames(cam, filepath = f"videos/{folder_name}/{fname}", iter_space = 30)
            except KeyboardInterrupt:
                raise
            except:
                print(f"{time.text} skipped")
            

