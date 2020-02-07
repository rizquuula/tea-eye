import cv2 
import numpy as np
import time

cam = cv2.VideoCapture(0)   # Open camera vision
num = 0
while True:     # Start looping the program
    ret, frame = cam.read()
    # Set color range for threshold
    min_color = np.array([86, 28, 181], dtype = "uint8")
    max_color = np.array([97, 99, 255], dtype = "uint8")
    
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # Converting BGR to HSV
    mask = cv2.inRange(HSV, min_color, max_color)   # Thresholding HSV to Binary color
    # Morphology Closing
    ImgErode = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))   # Morphology erode
    ImgDilate = cv2.dilate(ImgErode,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))   # Morphology dilate

    # Creating contour for detecting object
    contours, hierarchy = cv2.findContours(ImgDilate,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

    wh_box = [0]    # Saving w*h
    coordinateFull = [(0,0,0,0)]    # Saving the coordinate
    # Looping object with contour detection
    for cnt in contours:    
        x,y,w,h = cv2.boundingRect(cnt) # set the bounding of contour object
        # print("x,y,w,h : ",x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # Making bounding box for each object detected
        wh = w*h 
        wh_box.append(wh)
        coordinateFull.append((x,y,w,h))

    indexMaxBox = wh_box.index(max(wh_box))     # Set index of maximum bounding box size 
    x,y,w,h = coordinateFull[indexMaxBox]   # take the coordinate from index
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    # Cropping image from a largest bounding box
    cropImage = ImgDilate[y:y+h, x:x+w]
    # Clear the temporary box for saving 
    wh_box.clear()
    wh_box.append(0)
    coordinateFull.clear()
    coordinateFull.append((0,0,0,0))
    
    # Show everything start here
    cv2.imshow('Original Camera Vision', frame)
    # cv2.imshow('HSV Camera Vision', HSV)
    cv2.imshow('MASK Camera Vision', mask)
    # cv2.imshow('BLUR Camera Vision', blur)
    # cv2.imshow('NONOISE Camera Vision', nonoise)
    cv2.imshow('FILTERIMG Camera Vision', ImgDilate)
    
    # Show cropped image and resized one
    if (cropImage.shape[0] != 0) and (cropImage.shape[1] != 0):
        # print("showing")
        resized = cv2.resize(cropImage, (20,20), interpolation = cv2.INTER_AREA) 
        cv2.imshow('CROP Camera Vision', cropImage)
        cv2.imshow('CROP RESIZED Camera Vision', resized)
        print(num)
        num+=1

    # Key to close the app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("duatangan.jpg",frame)
        break
    
    # time.sleep(0.05)
    # time.sleep(1)

# Clear everything after program finished
cam.release()
cv2.destroyAllWindows()