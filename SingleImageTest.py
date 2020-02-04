import cv2 
import numpy as np

frame = cv2.imread("Example.png")

min_color = np.array([85, 40, 40], dtype = "uint8")
max_color = np.array([100, 250, 250], dtype = "uint8")

HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
mask = cv2.inRange(HSV, min_color, max_color)

ImgErode = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
ImgDilate = cv2.dilate(ImgErode,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))

contours, hierarchy = cv2.findContours(ImgDilate,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
wh_box = [0]
coordinateFull = [(0,0,0,0)]

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    # print("x,y,w,h : ",x,y,w,h)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(25,100,255),2)
    wh = w*h
    wh_box.append(wh)
    coordinateFull.append((x,y,w,h))

indexMaxBox = wh_box.index(max(wh_box))
x,y,w,h = coordinateFull[indexMaxBox]

cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

centerX = int(x+w/2)
centerY = int(y+h/2)

cv2.ellipse(frame,(centerX, centerY),(5,5),0,0,360,(0,0,255),2)
cv2.ellipse(frame,(centerX, centerY),(25,25),0,0,360,(0,0,255),2)

cropImage = ImgDilate[y:y+h, x:x+w]
wh_box.clear()
wh_box.append(0)
coordinateFull.clear()
coordinateFull.append((0,0,0,0))
cv2.imshow('Original Camera Vision', frame)
# cv2.imshow('HSV Camera Vision', HSV)
cv2.imshow('MASK Camera Vision', mask)
cv2.imshow('FILTERIMG Camera Vision', ImgDilate)
if (cropImage.shape[0] != 0) and (cropImage.shape[1] != 0):
    # print("show")
    resized = cv2.resize(cropImage, (20,20), interpolation = cv2.INTER_AREA) 
    cv2.imshow('CROP Camera Vision', cropImage)
    cv2.imshow('CROP RESIZED Camera Vision', resized)

cv2.imwrite("TeaEyeRGB.jpg",frame)
cv2.imwrite("TeaEyeHSV.jpg",HSV)
cv2.imwrite("TeaEyeMask.jpg",mask)
cv2.imwrite("TeaEyeErode.jpg",ImgErode)
cv2.imwrite("TeaEyeDilate.jpg",ImgDilate)
cv2.imwrite("TeaEyeCrop.jpg",cropImage)
cv2.imwrite("TeaEyeResized.jpg",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

