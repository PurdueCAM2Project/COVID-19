import requests
import os
from bs4 import BeautifulSoup
import re
import numpy as np
import datetime as dt
import urllib.request
from PIL import Image


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
class database_iterator():
    """
    Purpose: 
        a set of iterators to pull images from the http://vision.cs.luc.edu/~cv/images/ network cameras data base
    """

    def __init__(self, link = 'http://vision.cs.luc.edu/~cv/images/'):
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
        #Getting only the Folder directories 
        attrs = {
            'href': re.compile(r'5b[\w]+/')
        }
        folders = soup.find_all('a', attrs=attrs, string=re.compile(r'(?!\.)*$'))#only folder files
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
            'href': re.compile(r'([^\s]+(\.(?i)(jpg|png|gif|bmp))$)')
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
    
    def get_image(self, image, url = True, folder_name = None):
        """
        Purpose: 
            take a url or a path to an image and return an Pill image object
        
        input: 
            image: the url or path to the image
            url: False if the image is on the device(PATH) and not a url link to the data base
            folder_name: the name of the folder in the database to pull from

        yeild:  
            image: the image pill object  
        """
        if image == None:
            return None
        if folder_name != None:
            image = f"{self.link}{folder_name}{image['href']}"

        if url == True:
            image = Image.open(urllib.request.urlopen(image))
        else:
            image = Image.open(image)
        return image

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
