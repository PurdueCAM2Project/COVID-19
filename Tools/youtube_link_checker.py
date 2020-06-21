import requests
import os
import json 
import pandas as pd
import re

import pickle


"""
writers: 
    Vishnu Banna

Contact: 
    DM cam2 slack: 
        Vishnu Banna
    email: 
        vbanna@purdue.edu
"""

"""
need to enable the API in the google cloud platform: 
# move to team github

Installs:
    pip install --user pandas
    pip install --user numpy

TODO:
    documentation
    
"""



class validate():
    """
    Purpose: 
        Validate play lists of videos to ensure no duplicate links are used and all the links are live streams
    
    Raises: 
        Google API_KEY is not provided in init
            Exception("Google Cloud API Key Required to use the Youtube API")
        
        Path to exsting link is not provided, I will work on making this directly access our csv in the drive
            Exception("path to file containing current list of known links is required")
        
        Google has a request limit as a result it gives a "forbidden" reponse, just change this limit in the Google cloud console and you should be fine
            Exception("API Request Limit Exceeded, Change the Limit in Google Cloud")
        

    """
    def __init__(self, API_KEY = None, master_file_name = None):
        if (API_KEY == None):
            raise Exception("Google Cloud API Key Required to use the Youtube API")
#        if (master_file_name == None):
#            raise Exception("path to file containing current list of known links is required")
        self.API_KEY = API_KEY
        self.FILE_NAME = "https://docs.google.com/spreadsheets/d/1nWL72i2fj6NKcKOMbsNIm_RFT8KPj3z4BuTUFvZV3EA/export?format=csv&gid=1497128994"
        #load master sheet
        self.youtube_links = self.get_links()
        self.api_base = "https://www.googleapis.com/youtube/v3/"
        self.forbidden = False
        self.new_links = {}
        pass 

    def link_key(self, link):
        """
        Purpose: 
            given the link to the youtube video, return the video id key
        
        Inputs: 
            link (str): the link to the youtube video
        
        output: 
            (str): None if the link is not a youtube link else return the video id
        """
        match = re.search(re.compile("(youtube)"), link)
        if match == None:
            return str(None)
        red = link.split("?")[-1]
        red = re.findall(re.compile("v=[^&]+"), red)
        red = red[0].split("=")[-1] if len(red) > 0 else None
        return red

    def get_links(self):
        """
        Purpose: 
            take the provided CSV file and get the links for only the youtube videos
        
        Inputs: 
            none
        
        Returns: 
            a set with only the youtube video id keys 
        """
        df = pd.read_csv(self.FILE_NAME, header = None)
        df[9] = df[3].apply(lambda x : self.link_key(str(x)))
        links = set(df[9].values)
        links.remove("None")
        return links

    def __get_known_links(self):
        return

    def get_page(self, next_page, max_toks, id):
        """
        Purpose: 
            Make a request to the Youtube API for a certain page and return the appropriate infromation from the API Response
        
        Inputs: 
            next_page (str): the key for the next search page, found in reponse of an api call. if None, you are requesting the first page
            max_toks (int): the maximum tokens to retrive at once, ranging from (0, 50], the limit is set by google 
            id (str): the id of the playlist, extracted from the link
        
        Returns: 
            next_page (str): key for the next page
            prev_page (str): key for the previous page 
            total_items (int): total items in the page 
            items (list): list of videos retrived from the current search
            keys (list): list of keys including only live streams and unused links 
        """
        #identify the search page you are on and make the apporoiate api call
        if next_page == None:
            id = f"playlistItems?part=contentDetails%2C%20id%2C%20snippet%2C%20status&maxResults={max_toks}&playlistId={id}&key={self.API_KEY}"
        else:   
            id = f"playlistItems?part=contentDetails%2C%20id%2C%20snippet%2C%20status&maxResults={max_toks}&pageToken={next_page}&playlistId={id}&key={self.API_KEY}"
        items = requests.get(self.api_base + id)

        #check that the request is ok
        if not items.ok:
            print(f"\n{items.reason}, {items}\n")
            self.forbidden = True
            raise Exception("API Request Limit Exceeded, Change the Limit in Google Cloud")
        
        #convert request to json and return only the values we need 
        items = items.json()
        return self.__get_items(items)

    def __get_items(self, items):
        """
        Purpose: 
            Given response from self.get_page extract the important items

        Inputs: 
            items (dict): the response of the API playlist request
        
        Returns: 
            next_page (str): key for the next page
            prev_page (str): key for the previous page 
            total_items (int): total items in the page 
            items (list): list of videos retrived from the current search
            keys (list): list of keys including only live streams and unused links 
        """
        keys = []   
        try: 
            total_items = items["pageInfo"]["totalResults"]
        except:
            total_items = None
        
        try:
            next_page = items["nextPageToken"]
        except:
            next_page = None

        try:
            prev_page = items["prevPageToken"]
        except:
            prev_page = None

        try:
            for item in items["items"]:
                id = self.__validate_link(item)
                if id != None:
                    keys.append(id)
                print(id)  
        except:
            keys = keys
        keys = self.__check_livestream(keys)
        return next_page, prev_page, total_items, items, keys

    def __check_livestream(self, keys):
        """
        Purpose: 
            check to ensure the video link provided is a live stream
        
        Inputs: 
            id (str): the id of the video (found in the link) to validate
        
        Returns:
            id (str or None): return id if the link is a live stream, None if it is not
        """
        keys = list(keys)
        req_size = 50
        key_str = ",".join(keys)

        #print (key_str)
        id_search = f"videos?part=liveStreamingDetails&id={key_str}&key={self.API_KEY}"
        search = requests.get(self.api_base + id_search)

        #print(search.ok, search.reason)
        keys = []
        search_values = search.json()
        if search.ok :
          if "items" in search_values.keys():
            searches = search_values["items"]
            #print(type(searches))

          
            for item in searches:
                temp = self.check_livestream(item = item)
                if temp != None:
                    keys.append(temp)
        else:
          print(search.reason)
        return keys

    def check_livestream(self, item = None, id = None):
        if (type(item) == type(None)):
            print("soon to implement")
            id_search = f"videos?part=liveStreamingDetails&id={id}&key={self.API_KEY}"
            search = requests.get(self.api_base + id_search)
            search = search.json()
            item = search["items"][0].keys()
        
        #print(item)
        id = item["id"] 
        if "liveStreamingDetails" in item.keys():
            return id
        else:
            return None
        return

    def __validate_link(self, item):
        """
        Purpose: 
            given a video item json ensure that the link is not already being used, and it is a live stream 
        
        Inputs: 
            item (dict): the video details
        
        Returns:
            id (str or None): return id if the link is a live stream, None if it is not
        """
        timeout = 0
        id = str(item["contentDetails"]["videoId"])
        if str(id) in self.youtube_links:
            print("already used")
            return None
        return id #self.check_livestream(id)

    def playlist(self, url, save_links_to_file = False):
        """
        Purpose: 
           Given the url to a playlist, ensure all the videos are not all used and all of them are live streams
        
        Inputs: 
            url (str): the url to the playlist
            save_links_to_file (Bool): cunstruct a csv to store the links to a file, the file name is the <playlist ID>.csv
        
        Returns:
            if save_links_to_file:
                keys(str) : the name of the csv file 
            else: 
                keys(set) : the keys of all unique livestreams in the playlist
        """
        id = url.split("=")[-1]
        max_toks = 50
        keys = []

        first_key, prev_page, total_items, items, key  = self.get_page(None, max_toks, id)
        keys = key
        next_page, prev_page, total_items, items, key = self.get_page(first_key, max_toks, id)
        keys += key

        while next_page != first_key and next_page != None:
            next_page, prev_page, total_items, items, key = self.get_page(next_page, max_toks, id)
            keys += key

        keys = set(keys)

        if save_links_to_file and not self.forbidden: 
            return self.construct_csv(keys, id = id)
        return keys

    def construct_csv(self, keys, id = None):
        """
        Purpose: 
            given a set of keys and the playlist id, consturct a csv containing a link to all the video streams
        
        Inputs: 
            keys (set): a set of keys that need to be re constructed into youtube links
            id (str): the playlist id, if none is provided, uses the id of the first video in keys
        
        Returns: 
            (str): the name of the csv file 
        """
        keys = list(keys)
        if (id == None):
            id = keys[0]

        file = open(f"{id}.csv", "w")
        for key in keys:
            link = self.__construct_link(key)
            file.write(link)
        file.close()
        return f"{id}.csv"
    
    def add_link_to_csv(self):
        return

    def __construct_link(self, key):
        """
        Purpose: 
            given a key , consturct a link to the video associated with key
        
        Inputs: 
            keys (str):the key that needs to be reconstructed into a youtube link
            
        Returns: 
            (str): constucted link
        """
        base = "https://www.youtube.com/watch?v="
        return base + str(key) + ", \n"


v = validate(API_KEY=None).playlist("https://www.youtube.com/playlist?list=PLwygboCFkeeA2w1fzJm44swdG-NnyB6ip")
print(len(v))



"""
Notes : 
GET https://www.googleapis.com/youtube/v3/playlistItems?part=contentDetails%2C%20id%2C%20status&maxResults=[maxvids]&playlistId=[id]&key=[YOUR_API_KEY] HTTP/1.1
GET https://www.googleapis.com/youtube/v3/videos?part=liveStreamingDetails&id=KwFhEB34Hj8&key=[YOUR_API_KEY] HTTP/1.1


Authorization: Bearer [YOUR_ACCESS_TOKEN]
Accept: application/json
id = f"playlistItems?part=contentDetails%2C%20id%2C%20status&playlistId={id}&key={self.API_KEY}"
urlid = f"playlistItems?part=contentDetails%2C%20id%2C%20snippet%2C%20status&maxResults={max_toks}&playlistId={id}&key={self.API_KEY}"
tems = requests.get(api_base + urlid).json()
        
use next token [next_page] and previous token [prev_page] ID to itrate a playlist
use page token in the api URL to iterate play list

id = f"playlistItems?part=contentDetails%2C%20id%2C%20snippet%2C%20status&maxResults={max_toks}&pageToken={next_page}&playlistId={id}&key={self.API_KEY}"
->["items"][item_number]["contentDetails"]["videoId"] -> to get youtueb video link
"""
