# Estimating worldwide social distancing during COVID-19 through Computer Vision
### Generating Dataframes from Detection JSONs
From the repo's $HOME folder:

To generate Dataframes from a detection JSON when the day-night JSON is available:
```
python3 Analyze_All_Data/JSON_to_dataframe.py --image-json path_to_detection_JSON --day-night-json path_to_day_night_JSON --object person --cam-type image --savename filename_to_save --savedir directory_to_save_dataframes
```

The aforementioned command is valid when the object type concerned is people and the camera type concerned is video cameras. The object type can be changed to vehicles by changing the command line argument ```--object person``` to ```--object vehicle```. Likewise, the camera type can be changed by replacing ```--cam-type image``` with ```--cam-type video```.

To generate Dataframes from a detection JSON when the day-night JSON is not available:
```
python3 Analyze_All_Data/JSON_to_dataframe.py --image-json path_to_detection_JSON --object person --cam-type image --savename filename_to_save --savedir directory_to_save_dataframes
```
To generate the dataframes for object type vehicles or camera type video, changes similar to the case when day-night JSON is available can be followed.

### Plotting Results
In the Visualizations folder:

To plot scatterplots of a country or US state:

```
python3 scatterplots.py --country=country_name
python3 scatterplots.py --state=US_state_name
```

To include color coded date ranges (currently supports up to 5 date intervals):

```
python3 scatterplots.py --country=country_name --date1=='MM_DD ', --date2='MM_DD '
```

Dates must be in the format 'MM_DD_' including the trailing space. 

For example, 
```
python3 scatterplots.py --country=France --date1=='05-11 '
```

results in this plot:

<img width="599" alt="Screen Shot 2020-08-29 at 8 46 22 PM" src="https://user-images.githubusercontent.com/42527012/91648804-d40eda00-ea39-11ea-9d83-5ddd9bfba295.png">



### Checking For Social Distancing Violations

- our solution uses 2 metrics to determine violation: 
  1. size of bounding boxes to determine depth difference
  2. euclidean pixel distance 

We make the assumption that all bounding box heights are representative of an average person (5.4 feet), and all bounding boxes are true positives. More details can be found in the paper: 

Brief description of algorithm:
- the algorithm will first : 
  1. first detect people in a scene and determine a bounding box for each person 
- take 2 bounding boxes at a time and: 
  2. compute depth similarity, the relative size of the bounding boxes based on the bounding box height
  3. compute the euclidean pixel distance between centers of the boxes
  4. multiply the depth similarity and the euclidean pixel distance 
   -   boxes faraway from the camera have smaller bounding boxes, and object closer to the camera have larger bounding boxes 
   -   in our calculation, if 2 boxes have a similar height, then both must be closer to the camera
   -   if 2 boxes are very close, pixel distance will be low
  5. checking if this value is greater than 1.11 (6/5.4) determines the pairwise violation value
  
 <img width="1391" alt="Screen Shot 2020-08-29 at 8 56 09 PM" src="https://user-images.githubusercontent.com/42527012/91648818-133d2b00-ea3a-11ea-813b-b400f30c3c41.png">


### Collecting raw detection data

Create a set of cronjobs to call a bash script to call the following python files, or run these in the background at your convenience on a machine with GPU: 
  detect_cars_image_cams.py
  detect_people_image_cams.py
  detect_people_video_cams.py
  
  
 
