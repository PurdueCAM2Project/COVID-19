import cv2 as cv
import sys
import os


import counter
import camera_filter
import download


def main():
    #the path to the images of the camera 
    if len(sys.argv) != 2:
         print("Usage: python server.py [number of days]")
    day_threshold = int(sys.argv[1])
    images_path = '/home/shane/cam2/covid19/temp_camera_images'
    camera_list = counter.main(parent_link="http://vision.cs.luc.edu/~cv/images/", day_count=day_threshold, extensions_list=['png'])
    fileObject = open("report.txt", "w")
    fileObject.close()
    for camera in camera_list:
        #delete the files that currently inhabit the temp image folder 
        for filename in os.listdir(images_path):
            file_path = os.path.join(images_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        #download the images from the camera
        download.main(camera, images_path)
        #filter the images
        camera_filter.main(0, images_path, day_threshold)
        usable_count = len([name for name in os.listdir(os.path.join(images_path, 'usable_data'))])
        fileObject = open("report.txt", "a+")
        if usable_count < day_threshold:
            #unusable
            print(str(camera), "\nSTATUS: UNUSABLE\nREASON:\n\n")
        else:
            #usable
            print(str(camera), "\nSTATUS: USABLE\n\n")
        fileObject.close()
        





if __name__ == "__main__":
    main()
