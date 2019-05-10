import cv2
from PIL import Image
import numpy as np
import os
import math

# Playing video from file:
cap = cv2.VideoCapture('./data/train.mp4')

# number of frames
framesLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

neuronalTraining = []

print(framesLength)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0

rightReference = []
leftReference = []
lastPosRight = []
lastPosLeft = []

horizon = 260

speed = 0
lineSize = 3
speedData = []

# Read the template 
template = cv2.imread('./data/road-left/00.jpg', 0) 
template2 = cv2.imread('./data/road-right/00.jpg', 0) 


#template0 = cv2.imread('./data/tests/test1.jpg', 0) 
#template1 = cv2.imread('./data/tests/test2.jpg', 0) 

# Store width and heigth of template in w and h 
w, h = template.shape[::-1]
#w0, h0 = template0.shape[::-1]
#w1, h1 = template1.shape[::-1] 
w2, h2 = template2.shape[::-1] 

with open("./data/train.txt", "r") as filehandle:  
    for line in filehandle:
        # remove linebreak which is the last character of the string
        speed = line[:-1]

        # add item to the list
        speedData.append(speed)

def calcSlope(x1, x2, y1, y2):
    m = (y2 - y1) / (x2 - x1)
    return m

def calcN(m, x, y):
    # y = m * x + n
    # y - n = m * x
    # -n = (m * x) - y
    n = -1 * ((m * x) - y)
    return n

def getX(m, y, n):
    # y = m * x + n
    x = (y - n) / m
    return x


# Read until video is completed
while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
      
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    crop_img = thresh[250:370, 120:520]
    edges = cv2.Canny(crop_img, 75, 100)


    lines = cv2.HoughLinesP(edges,1, np.pi/180, 65, maxLineGap=25)
    if lines is not None:

        print(lines)

        for line in lines:
            x1, y1, x2, y2 = line[0]

            

            m = calcSlope(x1 + 120, x2 + 120, y1 + 250, y2 + 250)
            n = calcN(m, x1 + 120, y1 + 250)

            x11 = getX(m, 250, n)
            x21 = getX(m, 360, n)
           
            if(math.isinf(x11) or x11 < 0 or math.isnan(x11) & math.isinf(x21) or x21 < 0 or math.isnan(x21)):
                print('not number')
            else:

                print(x11, x21)

                cv2.line(frame, (int(x11), 250), (int(x21), 360), (255,0,0), 2)

            #cv2.line(frame, (x1 + 120, y1 + 250), (x2 + 120, y2 + 250), (0, 255, 0), 1)
            #cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.imwrite('./data/road-test/' + str(currentFrame) + '.jpg', crop_img)

    # horizon line
    #cv2.line(frame, (0, 240), (640, 240), (255,0,0), 1)

    #car line
    #cv2.line(frame, (0, 360), (640, 360), (255,0,0), 1)

    right = (0,0)
    left = (0,0)

    if len(lastPosRight) > 0:
        
        x1 = lastPosRight[0][0][0][0]
        y1 = lastPosRight[0][0][0][1]

        x2 = lastPosRight[0][0][1][0]
        y2 = lastPosRight[0][0][1][1]

        rx2 = lastPosRight[0][1][1][0]
        #ry2 = lastPosRight[0][1][1][1]

        m = calcSlope(x1, x2, y1, y2)
        n = calcN(m, rx2, 360)
        x = getX(m, horizon, n)

        #cv2.line(frame, (int(x), horizon), (rx2, 360), (255,0,0), 2)
        
    if len(lastPosLeft) > 0:

        lx1 = lastPosLeft[0][0][1][0]
        ly1 = lastPosLeft[0][0][0][1]

        lx2 = lastPosLeft[0][1][0][0]
        ly2 = lastPosLeft[0][1][1][1]

        #cv2.line(frame, (lx1, 240), (lx2, 360), (255,0,0), 2)


        # x1 = lastPosLeft[0][0][0][0]
        # y1 = lastPosLeft[0][0][0][1]

        # x2 = lastPosLeft[0][0][1][0]
        # y2 = lastPosLeft[0][0][1][1]

        rx2 = lastPosLeft[0][1][1][0]
        # #ry2 = lastPosRight[0][1][1][1]

        m = calcSlope(lx1, lx2, ly1, ly2)
        n = calcN(m, rx2, 360)
        x = getX(m, horizon, n)

        #cv2.line(frame, (lx1, horizon), (lx2, 360), (255,0,0), 2)

        # print(lastPosLeft)

        # x1 = lastPosLeft[0][0][1][0]
        # y1 = lastPosLeft[0][0][0][1]

        # x2 = lastPosLeft[0][0][0][0]
        # y2 = lastPosLeft[0][0][1][1]

        # lx2 = lastPosLeft[0][1][0][0]
        # #ry2 = lastPosRight[0][1][1][1]

        # print(x1, y1, x2, y2, lx2)
        # print('')

        # m = calcSlope(x1, x2, y1, y2)
        # n = calcN(m, rx2, 360)
        # x = getX(m, 240, n)

        # cv2.line(frame, (int(x), 240), (lx2, 360), (255,0,0), 2)

    # Perform match operations. 
    res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED) 
    #res0 = cv2.matchTemplate(thresh, template0, cv2.TM_CCOEFF_NORMED) 
    #res1 = cv2.matchTemplate(thresh, template1, cv2.TM_CCOEFF_NORMED) 
    res2 = cv2.matchTemplate(thresh, template2, cv2.TM_CCOEFF_NORMED) 

    # Specify a threshold 
    threshold = 0.46

    # Store the coordinates of matched area in a numpy array 
    loc = np.where( res >= threshold)
    #loc0 = np.where( res0 >= threshold)
    #loc1 = np.where( res1 >= threshold)
    loc2 = np.where( res2 >= threshold) 

    locLength = len(loc[0]) > 0 and len(loc[1]) > 0
    #locLength0 = len(loc0[0]) > 0 and len(loc0[1]) > 0
    #locLength1 = len(loc1[0])  > 0 and len(loc1[1]) > 0
    locLength2 = len(loc2[0])  > 0 and len(loc2[1]) > 0

    if locLength:

        x1 = int(loc[1].min())
        y1 = int(loc[0].min())

        x2 = x1 + w
        y2 = y1 + h

        leftReference.append({ "frame": currentFrame, "position": [(x1, y1), (x2, y2)] })

        #cv2.line(frame, ( int(loc[1].mean() + w), int(loc[0].mean()) ), ( int(loc[1].mean()), int(loc[0].mean() + h)), (255,0,0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,235,0), 1) 
        
    else:
        if(len(leftReference) != 0):

            framesNum = len(leftReference)

            lastPosLeft = []
            lastPosLeft.append([leftReference[0]['position'], leftReference[framesNum - 1]['position']])

        leftReference = []

    #if locLength0:
        #cv2.line(frame, (0, loc[0][len(loc[0]) - 1]), (640, loc[0][len(loc[0]) - 1]), (0,0,255), 1)
        #cv2.line(frame, (0, loc[0][0] + h), (640, loc[0][0] + h), (0,0,255), 1)
        #cv2.rectangle(frame, (loc0[1][len(loc0[1]) - 1], loc0[0][len(loc0[0]) - 1]), (loc0[1][0] + w0, loc0[0][0] + h0), (0,0,255), 1) 

    #if locLength1:
        #cv2.line(frame, (0, loc1[0][len(loc1[0]) - 1]), (640, loc1[0][len(loc1[0]) - 1]), (0,255,0), 2)
        #cv2.line(frame, (0, loc1[0][0] + h1), (640, loc1[0][0] + h1), (0,255,0), 2)
        #cv2.rectangle(frame, (loc1[1][len(loc1[1]) - 1], loc1[0][len(loc1[0]) - 1]), (loc1[1][0] + w1, loc1[0][0] + h1), (255,0,0), 1) 

    #RIGHT REFERENCE
    if locLength2:
        
        x1 = int(loc2[1].min())
        y1 = int(loc2[0].min())

        x2 = x1 + w2
        y2 = y1 + h2

        rightReference.append({ "frame": currentFrame, "position": [(x1, y1), (x2, y2)] })
        
        #cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        #cv2.line(frame, (0, loc2[0][0] + h2), (640, loc2[0][0] + h2), (255,0,0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,235,0), 1) 

    else:

        if(len(rightReference) != 0):

            framesNum = len(rightReference)

            initialLBCPosition = rightReference[0]['position'][0]
            initialRTCPosition = rightReference[0]['position'][1]

            finalLBCPosition = rightReference[framesNum - 1]['position'][0]
            finallRTCPosition = rightReference[framesNum - 1]['position'][1]

            lastPosRight = []
            lastPosRight.append([rightReference[0]['position'], rightReference[framesNum - 1]['position']])
            

            # number of frames / startPos / finalPos / length of the line detected
            neuronalTraining.append({ 
                "frames": framesNum, 
                "startPos": [initialLBCPosition, initialRTCPosition], 
                "finalPos": [finalLBCPosition, finallRTCPosition], 
                "speed": speedData[currentFrame]
            })

            #print(neuronalTraining)
        rightReference = []


    #print('')
    # Display the resulting frame
    cv2.imshow('Detected', frame) 
    
    #cv2.waitKey()
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