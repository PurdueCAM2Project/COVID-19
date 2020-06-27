#usage "python filter.py [multiple cameras 1 or 0 (yes or no)] [directory path] [minimum number of days]" where the first argument is a 1 or a 0, 1 denoting that the [directory path] should be treated as a parent directory containing folders each subfolder denoting a single camera, a 0 denotes that [directory path] is the path to the folder of a single camera [directory path] is replaced by the path of the directory you want to iterate over and number of days is the minimum number of days that a camera has to have usable data for to be considered viable 

import cv2 as cv
import sys
import os
import numpy as np



def main(multiple_cameras=0, parent_directory_path='/', min_data_count_threshold=0):
    # ensure that all arguments are present
    if len(sys.argv) == 4:
        print("Usage: python filter.py [multiple cameras 1 or 0] [directory path] [number of days]")
        
        multiple_cameras = int(sys.argv[1]) #1 or 0 denoting whether or not the script is being called for multiple cameras
        parent_directory_path = sys.argv[2]
        min_data_count_threshold = int(sys.argv[3])

    if (multiple_cameras == 0):
        # we're only analyzing one camera
        inspect_camera(parent_directory_path, min_data_count_threshold)
    else:
        # we're analyzing multiple cameras, so we should iterate over all the subdirectories contained in the current directory
        total_viable_cameras = 0
        total_unviable_cameras = 0
        for dirname in os.listdir(parent_directory_path):
            if os.path.isdir(os.path.join(parent_directory_path, dirname)):
                viable = inspect_camera(os.path.join(parent_directory_path, dirname), min_data_count_threshold)
                if viable:
                    total_viable_cameras += 1
                else:
                    total_unviable_cameras += 1
        print("Number of viable cameras: ", total_viable_cameras)
        print("Number of unviable cameras: ", total_unviable_cameras)
    

                    
def inspect_camera(directory_path, min_data_count_threshold):
    # set up the necessary directories/ housekeeping stuff
    usable_dir_path = os.path.join(directory_path, "usable_data") #using os.path.join so the trailing / doesn't matter 
    unusable_dir_path = os.path.join(directory_path, "unusable_data")
    if not os.path.isdir(usable_dir_path):
        os.mkdir(usable_dir_path)
    if not os.path.isdir(unusable_dir_path):
        os.mkdir(unusable_dir_path)

    #iterate over the directory for images
    for filename in os.listdir(directory_path):
        #TODO: find out what the image format is
        if filename.endswith(".jpg") or filename.endswith(".png"): 
            image = cv.imread(os.path.join(directory_path, filename), 0)
            if check_img(image):
                #this image has passed the check and can be moved to the usable directory 
                os.rename(os.path.join(directory_path, filename),
                          os.path.join(usable_dir_path, filename))
            else:
                #this image is not usable, and therefore must be moved
                os.rename(os.path.join(directory_path, filename),
                          os.path.join(unusable_dir_path, filename))
    
    #iterate over the directory for videos
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp4"):
            cap = cv.VideoCapture(os.path.join(directory_path, filename))
            if (cap.isOpened() == False):
                print("Error with opening the video file ", os.path.join(directory_path,filename))
                continue
            frame_count = cap.get(7) # i think 7 is the correct property id TODO: verify this
            cap.set(1, 0)
            ret1, first_frame = cap.read()
            cap.set(1, frame_count/2)
            ret2, middle_frame = cap.read()
            cap.set(1, frame_count - 1)
            ret3, last_frame = cap.read()
            if ((not ret1) or (not ret2) or (not ret3)):
                #this image is corrupted and passed the check and can be moved to the unusable directory 
                os.rename(os.path.join(directory_path, filename),
                          os.path.join(unusable_dir_path, filename))
                continue

            video_usable = check_img(first_frame) and check_img(middle_frame) and check_img(last_frame)
            if video_usable:
                #this image has passed the check and can be moved to the usable directory 
                os.rename(os.path.join(directory_path, filename),
                          os.path.join(usable_dir_path, filename))
            else:
                #this image is not usable, and therefore must be moved
                os.rename(os.path.join(directory_path, filename),
                          os.path.join(unusable_dir_path, filename))
                
    #make sure the amount of images and videos are above 50%
    if (len(os.listdir(usable_dir_path)) < min_data_count_threshold):
        print("Camera data in ", directory_path, " is not viable")
        return False
    else:
        print("Camera data in ", directory_path, " is viable")
        return True

def check_img(image): #this image check function is incomplete TODO: implement further image scrutiny
    #validate the image using opencv to see if the image is completely white or black
    if np.mean(image) == 0:
        #this image is all black 
        return False
    elif np.mean(image) == 255: #the average can only be 255 if all the pixels are white
        #this image is all white
        return False
    #to determine if the image is blurry see if it is a gaussian blur or laplacian blur
    #to subvert the fact that many many of the images have text alongside the border of the image, we will only analyze the middle of it
    dimensions = image.shape
    height = dimensions[0]
    width = dimensions[1]
    cropped_image = np.arange(1)
    cropped_image = cropped_image.reshape(height/2, width/2)
    for i in range(height/2):
        for j in range(width/2):
            cropped_image[i][j] = image[i + (height/2)][j + (width/2)]
    beforeGaussianBlurValue = cv.Laplacian(cropped_image, cv.CV_64F).var()
    blurred_image = cv.blur(cropped_image, (5,5)) #this kernel might not be optimal but as it stands now this one works well enough as far as i know TODO: either verify that this is the best kernel or find a better one
    afterGaussianBlurValue = cv.Laplacian(blurred_image, cv.CV_64F).var()
    if afterGaussianBlurValue >= beforeGaussianBlurValue:
        #this image is too blurry, we know this because a gaussian blur applied on an image that isn't already blurry should decrease the variance value, but if it doesn't (it's more than or equal to) then the image is too blurry
        return False
    return True

def determine_day_night(image): #determines whether or not an image is captured during the day or night
    #0 denotes night, 1 denotes day
    if np.mean(image) > 60:
        #this image is all black 
        return False
    return 



if __name__ == "__main__":
    main()
