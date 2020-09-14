import cv2
import numpy as np

frameWidth = 720
frameHeight = 480
url = "http://192.168.0.102:8080/video"
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
mycolors = [#[90, 111, 108, 123, 255, 255], #blue
            #[127 ,115 ,118 ,179 ,195 ,255], #red
            #[132 ,43 ,133 ,179 ,183 ,255],
            [160 ,52 ,163 ,179 ,106 ,255]]#pink
            #[140 ,124 ,165 ,179 ,255 ,255]] #green
mycolorvalues = [[130, 79, 235]
                 
                 #[0, 0, 0],
                 #[0, 0, 0],
                 #[0, 0, 0]
                 ]

mypoints = []

def findcolors(im, mycolors,mycolorvalues):
    imgHsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    count  = 0
    newPoints = []
    for color in mycolors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHsv, lower, upper)
        x,y = getContours(mask)
        cv2.circle(imgResult, (x,y), 10, mycolorvalues[count], cv2.FILLED)
        if x != 0 and y!= 0:
            newPoints.append([x, y, count])
        count +=1
    return newPoints
        

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True) 
            x,y,w,h = cv2.boundingRect(approx)
    return x+w//2, y


def drawOnCanvas(mypoints, mycolorvalues):
     for point in mypoints:
         cv2.circle(imgResult, (point[0], point[1]), 10, mycolorvalues[point[2]], cv2.FILLED) 
while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findcolors(img, mycolors, mycolorvalues)
    if len(newPoints)!= 0:
        for newP in newPoints:
            mypoints.append(newP)
    if len(mypoints)!=0:
        drawOnCanvas(mypoints, mycolorvalues)
        
    findcolors(img, mycolors,mycolorvalues)
    imgResult = cv2.flip(imgResult, 1)
    cv2.imshow('Horizontal Stacking', imgResult) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
