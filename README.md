# COVID-19


### Plotting Results
In the Violations folder:

To plot scatterplots of a country or US state:

```
python3 --country=country_name
python3 --state=US_state_name
```

To include color coded date ranges (currently supports up to 5 date intervals):

```
python3 --country=country_name --date1=='MM_DD ', --date2='MM_DD '
```

Dates must be in the format 'MM_DD_' including the trailing space. 

For example, 
```
python3 --country=France --date1=='05-11 '
```

results in this plot:




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
  
 
