# COVID-19



### Checking For Social Distancing Violations

- our solution uses 2 metrics to determine violation: 
  1. object perspective
  2. euclidean pixel distance 

- the algorithm will first : 
  1. first detect people in a scene and determine a bounding box for each person 
- take 2 bounding boxes at a time and: 
  2. compute perspective similarity, the relative size of the bounding boxes based on the area
  3. compute the euclidean pixel distance between centers of the boxes
  4. multiply the perspective similarity and the euclidean pixel distance 
         -   boxes faraway from the camera have smaller bounding boxes, and object closer to the camera have larger bounding boxes 
         -   in our calculation, if 2 boxes have a similar area, then both must be closer to the camera
         -   if 2 boxes are very close, pixel distance will be low
         -   by multiplying perspective similarity and euclidean pixel, you are telling the algorithm how much weight to apply to the euclidean distance when calculating the social distance violation 
  5. finally, we use the height of the largest bounding box to scale the multiplication of the perspective similarity and the euclidean pixel distance by the height of the tallest person in out comparison. this allows us to convert relative pixel distance to an estimation of true distance determined by the average height of a a person. 
