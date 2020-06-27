#usage python counter.py [link to the parent directory webpage] [day count (minimum number of] [extension that you want to search for, example: .jpg, .mp4, etc]
#a lot of this code comes from https://www.thepythoncode.com/article/extract-all-website-links-python

import lxml
from lxml import html
import requests
import urllib
from requests_html import HTMLSession
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
#import colorama
import sys


external_urls = set()
internal_urls = set()


def main(parent_link = '/', day_count=0, extensions_list=['png']):
    """
if len(sys.argv) >= 4:
        print("Usage: python counter.py [link to the parent directory webpage] [extension that you want to search for, example: .jpg, .mp4, etc]")

    parent_link = sys.argv[1]
    day_count = int(sys.argv[2])
    extensions_list = []
    for i in range(3, len(sys.argv)): #populate the list of extensions that count as acceptable entries
      extensions_list.append(str(sys.argv[i]))
    """
    return count(get_all_cameras(parent_link), extensions_list, day_count)
    #this main function is really gross, it's not even usable independently in it's current state, it has to be called from server.py TODO: FIX this
    

    
def count(urls, extensions_list, day_count):
    #link links to a child directory, this function
    total_camera_count = len(urls)
    feasible_camera_count = 0 
    feasible_camera_list = []
    urls = set(urls)
    for camera in urls:
        total_camera_count += 1
        fp = urllib.request.urlopen(camera)
        mybytes = fp.read()
        html_str = mybytes.decode("utf8")
        fp.close()
        entry_count = 0
        for extension in extensions_list:
            entry_count += html_str.count(str(extension))
        if entry_count >= day_count:
            #the camera is feasible and can be counted
            feasible_camera_count += 1
            feasible_camera_list.append(str(camera))
            print(str(camera))
    print("Total cameras: ", total_camera_count)
    print("Feasible cameras: ", len(set(feasible_camera_list)))
    return feasible_camera_list
    
def is_valid(url):
    if not (str(url)[-1] == '/'):
        return False
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_cameras(url):
    """
    Returns all URLs that is found on `url` in which it belongs to the same website
    """
    # all URLs of `url`
    urls = set()
    # domain name of the URL without the protocol
    domain_name = urlparse(url).netloc
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            # href empty tag
            continue
        # join the URL if it's relative (not absolute link)
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        # remove URL GET parameters, URL fragments, etc.
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if not is_valid(href):
            # not a valid URL
            continue
        if href in internal_urls:
            # already in the set
            continue
        if domain_name not in href:
            # external link
            if href not in external_urls:
                external_urls.add(href)
            continue
        urls.add(href)
        internal_urls.add(href)
    return urls




if __name__ == "__main__":
    main()
