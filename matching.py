# Python program to illustrate 
# template matching 
import cv2 
import numpy as np 

# Read the main image 
img_rgb = cv2.imread('./data/frames/frame-train-1.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_gray, (5, 5), 10)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Read the template 
template = cv2.imread('./data/frames/save4.jpg',0) 

# Store width and heigth of template in w and h 
w, h = template.shape[::-1] 

# Perform match operations. 
res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED) 

# Specify a threshold 
threshold = 0.49

# Store the coordinates of matched area in a numpy array 
loc = np.where( res >= threshold) 

# Draw a rectangle around the matched region. 
for pt in zip(*loc[::-1]): 
	cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 

# Show the final image with the matched area. 
cv2.imshow('Detected', img_rgb) 
cv2.waitKey(0)