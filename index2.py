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
    #blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret3,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    crop_img_left = thresh[280:370, 80:230]
    crop_img_right = thresh[280:370, 380:560]

    cv2.imwrite('./data/road-test-left/' + str(currentFrame) + '-left.jpg', crop_img_left)
    cv2.imwrite('./data/road-test-rigth/' + str(currentFrame) + '-right.jpg', crop_img_right)

    # Read the template 
    template = cv2.imread('./data/tests/left-1.png', 0) 
    template1 = cv2.imread('./data/tests/left-2.png', 0) 
    template2 = cv2.imread('./data/tests/test1.jpg', 0) 


    #template = cv2.imread('./data/tests/test1.jpg', 0) 
    #template1 = cv2.imread('./data/tests/test2.jpg', 0) 

    # Store width and heigth of template in w and h 
    w, h = template.shape[::-1]
    w1, h1 = template1.shape[::-1] 
    w2, h2 = template2.shape[::-1] 

    # Perform match operations. 
    res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED) 
    res1 = cv2.matchTemplate(thresh, template1, cv2.TM_CCOEFF_NORMED) 
    res2 = cv2.matchTemplate(thresh, template2, cv2.TM_CCOEFF_NORMED) 

    # Specify a threshold 
    threshold = 0.59

    # Store the coordinates of matched area in a numpy array 
    loc = np.where( res >= threshold)
    loc1 = np.where( res1 >= threshold)
    loc2 = np.where( res2 >= threshold) 

    locLength = len(loc[0])
    locLength1 = len(loc1[0])
    locLength2 = len(loc2[0])

    #cv2.line(frame, (0, 360), (640, 360), (0,255,255), 1)

    #if locLength > 0:
        
        #cv2.line(frame, (0, loc[0][len(loc[0]) - 1]), (640, loc[0][len(loc[0]) - 1]), (0,0,255), 1)
        #cv2.line(frame, (0, loc[0][0] + h), (640, loc[0][0] + h), (0,0,255), 1)
        #cv2.rectangle(frame, (loc[1][len(loc[1]) - 1], loc[0][len(loc[0]) - 1]), (loc[1][0] + w, loc[0][0] + h), (0,0,255), 1) 
    
    #if locLength1 > 0:
        
        #cv2.line(frame, (0, loc1[0][len(loc1[0]) - 1]), (640, loc1[0][len(loc1[0]) - 1]), (0,255,0), 2)
        #cv2.line(frame, (0, loc1[0][0] + h1), (640, loc1[0][0] + h1), (0,255,0), 2)
        #cv2.rectangle(frame, (loc1[1][len(loc1[1]) - 1], loc1[0][len(loc1[0]) - 1]), (loc1[1][0] + w1, loc1[0][0] + h1), (0,255,0), 2) 




    #RIGHT REFERENCE
    if locLength2 > 0:
        rightReference.append(currentFrame)
        cv2.line(frame, (0, loc2[0][len(loc2[0]) - 1]), (640, loc2[0][len(loc2[0]) - 1]), (255,0,0), 2)
        cv2.line(frame, (0, loc2[0][0] + h2), (640, loc2[0][0] + h2), (255,0,0), 2)
        cv2.rectangle(frame, (loc2[1][len(loc2[1]) - 1], loc2[0][len(loc2[0]) - 1]), (loc2[1][0] + w2, loc2[0][0] + h2), (255,0,0), 2) 
    else:

        if(len(rightReference) != 0):
            timeSg = len(rightReference) / 20
            speed = (3 / 1000) / (timeSg / 3600)

            speedKmh = str(round(speed, 2))
            speedMph = str(round(speed / 1.609, 2))

            print(currentFrame)
            print( speedKmh + 'km/h')
            print( speedMph + ' mph')
            print('')

        rightReference = []



    #print('')
    # Display the resulting frame
    cv2.imshow('Detected', crop_img) 
    
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
    currentFrame += 1

  # Break the loop
  else: 
    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()