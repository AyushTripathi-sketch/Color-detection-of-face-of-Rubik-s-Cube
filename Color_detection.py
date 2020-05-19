import cv2
import numpy as np

#capturing video through webcam
cap=cv2.VideoCapture(0)

while True:
    ret, img = cap.read()                    #read the image through webcam
    if not ret:
        print("failed to grab the frame")
        break
    

    '''scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img= cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('original',img)'''


        

    #converting frame(img i.e BGR) to HSV(Hue-Saturation-Value)

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv format',hsv)

    #defining the range of red Colour
    red_lower=np.array([167,100,100],dtype=np.uint8)
    red_upper=np.array([180,255,255],dtype=np.uint8)

    #defining the range of the blue color
    blue_lower=np.array([97,100,100],dtype=np.uint8)
    blue_upper=np.array([103,255,255],dtype=np.uint8)

    #defining the range of yellow color
    yellow_lower=np.array([24,100,100],dtype=np.uint8)
    yellow_upper=np.array([37,255,255],dtype=np.uint8)

    #defining the range of green color
    green_lower=np.array([42,100,100],dtype=np.uint8)
    green_upper=np.array([75,255,255],dtype=np.uint8)

    #defining the range of orange color
    orange_lower=np.array([10,100,100],dtype=np.uint8)
    orange_upper=np.array([16,250,250],dtype=np.uint8)

    #defining the range of white color
    white_lower=np.array([70,10,130],dtype=np.uint8)
    white_upper=np.array([180,110,255],dtype=np.uint8)

    #finding the range of red,blue,white,orange and yellow colour
    red=cv2.inRange(hsv,red_lower,red_upper)
    blue=cv2.inRange(hsv,blue_lower,blue_upper)
    yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
    green=cv2.inRange(hsv,green_lower,green_upper)
    orange=cv2.inRange(hsv,orange_lower,orange_upper)
    white=cv2.inRange(hsv,white_lower,white_upper)
        
    #Morphological transfoorrmation ,Dilation
    kernal = np.ones((5,5), "uint8")

    red = cv2.dilate(red,kernal)
    res = cv2.bitwise_and(img, img, mask = red)

    blue = cv2.dilate(blue,kernal)
    res1 = cv2.bitwise_and(img, img, mask = blue)

    yellow = cv2.dilate(yellow,kernal)
    res2 = cv2.bitwise_and(img, img, mask = yellow)

    green = cv2.dilate(green,kernal)
    res3 = cv2.bitwise_and(img, img, mask = green)

    orange = cv2.dilate(orange,kernal)
    res4 = cv2.bitwise_and(img, img, mask = orange)

    white = cv2.dilate(white,kernal)
    res5 = cv2.bitwise_and(img, img, mask = white)

    #Tracking the Red Colour
    (contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area>300):

            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,"RED",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    #Tracking the Blue Colour
    (contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area>300):

            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"BLUE",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    #Tracking the Yellow Colour
    (contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area>300):

            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,"Yellow",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))
        
    #Tracking the Green Colour
    (contours,hierarchy)=cv2.findContours(green,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area>300):

            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"Green",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))


    #Tracking the orange Colour
    (contours,hierarchy)=cv2.findContours(orange,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area>300):

            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"Orange ",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    #Tracking the White Colour
    (contours,hierarchy)=cv2.findContours(white,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area>300):

            x,y,w,h = cv2.boundingRect(contour)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"White",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255))

    cv2.imshow("Color Tracking",img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        cap.release()
        cv2.destroyAllWindows()
        break

