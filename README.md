# canny-edge-detection
This is canny edge detection from scratch in openCV and python . <br>
The Canny edge detection algorithm is composed of 5 steps:<br>
  1. Noise reduction;
  2. Gradient calculation;
  3. Non-maximum suppression;
  4. Double threshold;
  5. Edge Tracking by Hysteresis.


## Input
![This is an image](https://github.com/nahid0335/canny-edge-detection/blob/main/tiger2.jpg)
<br>
convert it into gray scale - <br>
![This is an image](https://github.com/nahid0335/canny-edge-detection/blob/main/1.PNG)

## Noise reduction
use gaussian blur to smooth the image and remove noise . <br>
kernel size = 5<br>
sigma = 1 <br>
after bluring the output will be - <br>
![This is an image](https://github.com/nahid0335/canny-edge-detection/blob/main/2.PNG)

## Gradient calculation
use sobel filter to calculate the gradient derivation.<br>
kernel size of sobel filter is 3.<br>
![This is an image](https://github.com/nahid0335/canny-edge-detection/blob/main/3.PNG)

## Non-maximum suppression
to make the edge thin , we need to use no maximum suppression.<br>
![This is an image](https://github.com/nahid0335/canny-edge-detection/blob/main/4.PNG)

## Double threshold
using double threshold, we will eleminate some weak edge.<br>
lowThresholdRatio = 0.05<br>
highThresholdRatio = 0.15<br>
weak pixel = 25<br>
strong pixel = 255<br>
![This is an image](https://github.com/nahid0335/canny-edge-detection/blob/main/5.PNG)

## Edge Tracking by Hysteresis(Output)
Based on the threshold results, the hysteresis consists of transforming weak pixels into strong ones, 
if and only if at least one of the pixels around the one being processed is a strong one.<br>
finally the output - <br>
![This is an image](https://github.com/nahid0335/canny-edge-detection/blob/main/6.PNG)
