import cv2
from PIL import Image
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('./data/train.mp4')

# number of frames
framesLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(framesLength)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0

rightReference = []

speed = 0
lineSize = 3

# Read until video is completed
while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
      
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    ret3,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    crop_img_left = thresh[250:400, 80:230]
    crop_img_right = thresh[250:400, 380:560]

    cv2.imwrite('./data/road-left/' + str(currentFrame) + '-left.jpg', crop_img_left)
    cv2.imwrite('./data/road-right/' + str(currentFrame) + '-right.jpg', crop_img_right)

    currentFrame += 1

  # Break the loop
  else: 
    break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()