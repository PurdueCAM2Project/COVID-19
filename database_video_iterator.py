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
writers: 
    Vishnu Banna
    Mohammed Metwaly

Contact: 
    DM cam2 slack: 
        Vishnu Banna
    email: 
        vbanna@purdue.edu
        mmetwaly@purdue.edu
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
        
    def get_all_frames(self, image, url = True, folder_name = None):
        # name="http://vision.cs.luc.edu/~cv/videos/bcfe3110b5/2020-05-11_23_54_52.mp4"
        print(folder_name)
        if image == None:
            return None
        if folder_name != None:
            image = f"{self.link}/{folder_name}/{image['href']}"
        # frames = []
        #image = imageio.get_reader(image)
        #for i, frame in enumerate(image):
        #frames = list(image)
        #del image
        vidcap = cv2.VideoCapture(image)
        success, frame = vidcap.read()
        count = 0
        while(success):
             cv2.imwrite("frame%d.png" % count, frame)
             success, image = vidcap.read()
             count+=1
        return frames

    def get_arbitrary_images(self, get_img = False):
        """
        ITERATOR
        --------
        Purpose: 
            Iterate over all cameras and return the link to an arbitrary images in each folder
        
        input: 
            get_img: if (True), retruns a PIL image object, if False, returns links to an image -> call self.get_image(url) to get the image
            True is not recomended as it is very slow
        
        yeild: 
            folder_name: the name of the folder you are currently pulling an image from 
            images: the arbitrary image url or pill object
            time: the date and time as a string
        """
        for folder in self.folders:
            folder_name = folder["href"]

            self.imgs = self.get_images(folder_name)
            self.dtms = self.get_datetimes(folder_name)

            print(len(self.imgs), len(self.dtms))
            if len(self.imgs) > 0:
                self.random_select = np.random.randint(0, len(self.imgs))
                link = f"{self.link}{folder_name}{self.imgs[self.random_select]['href']}"
                time = self.dtms[self.random_select].text#dt.datetime.strptime(str(dtms[imgselect].text), '%y-%m-%d %H:%M  ') 
                if get_img:
                    yield folder_name, self.get_image(link), time
                else:
                    yield folder_name, link, time
            else:
                self.random_select = None
                yield f"{folder_name} -> empty", None, None

    def get_n_arbitrary_images(self, num_rand = 2):
        """
        ITERATOR
        --------
        Purpose: 
            Iterate over all cameras and return the link to an arbitrary images in each folder
        
        input: 
            get_img: if (True), retruns a PIL image object, if False, returns links to an image -> call self.get_image(url) to get the image
            True is not recomended as it is very slow
        
        yeild: 
            folder_name: the name of the folder you are currently pulling an image from 
            images: the arbitrary image url or pill object
            time: the date and time as a string
        """
        for folder in self.folders:
            folder_name = folder["href"]

            self.imgs = self.get_images(folder_name)
            self.dtms = self.get_datetimes(folder_name)

            print(len(self.imgs), len(self.dtms))
            if num_rand > len(self.imgs):
                links = []
                for image in self.imgs:
                    links.append(f"{self.link}{folder_name}{image['href']}")
                yield folder_name, links, self.dtms
            elif len(self.imgs) > 0:
                #self.random_select = np.random.randint(0, len(self.imgs), num_rand)
                self.random_select = np.sort(np.random.choice(len(self.imgs), num_rand, replace=False))
                #print(self.random_select)
                links = []
                times = []
                for index in self.random_select:
                    links.append(f"{self.link}{folder_name}{self.imgs[index]['href']}")
                    times.append(self.dtms[index].text)
                yield folder_name, links, times
            else:
                self.random_select = None
                yield f"{folder_name} -> empty", None, None
    
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
    
    def get_all_images(self, get_img = False):
        """
        ITERATOR
        --------
        Purpose: 
            Iterate over all cameras and return the link to all images in each folder
        
        input: 
            get_img: if (True), retruns a list of PIL image object, if False, returns links to each image -> call self.get_image(url) to get the image
            True is not recomended as it is very slow
        
        yeild: 
            folder_name: the name of the folder you are currently pulling images from 
            images: the list of images urls or pill objects
            self.dtms: the time paralleling each image in the images as a beautiful soup object, call object.text to get the date and time as a string
        """
        for folder in self.folders:
            folder_name = folder["href"]

            self.imgs = self.get_images(folder_name)
            self.dtms = self.get_datetimes(folder_name)

            print(len(self.imgs), len(self.dtms))
            if get_img:
                images = []
                for image in self.imgs:
                    link = f"{self.link}{folder_name}{image['href']}"
                    images.append(self.get_image(link))
                yield folder_name, images, self.dtms
            else:
                links = []
                for image in self.imgs:
                    links.append(f"{self.link}{folder_name}{image['href']}")
                yield folder_name, links, self.dtms



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
            #try:
            if cam != None:
                image = k.get_all_frames(cam)
                if image != None:
                    fname = cam.split("/")[-1].split(".")[0]
                    os.mkdir(f"videos/{folder_name}/{fname}")
                    print(len(image))
                    for i in range(0, len(image), 10):
                        plt.imsave(f"videos/{folder_name}/{fname}/image_{i}.png", image[i])
            # except:
            #     print(f"{time.text} skipped")
            

        os.system(f"zip {folder_name}.zip {folder_name}/*")
        os.system(f"rm -r {folder_name}/*")
        os.system(f"rmdir {folder_name}")
