import cv2 
import numpy as np
import time

cam = cv2.VideoCapture("VideoTenThousandCave.avi")
# cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 3)

startcam = True
num = 0

while startcam:
    ret, frame = cam.read()
    #Light
    # min_color = np.array([85, 10, 200], dtype = "uint8")
    # max_color = np.array([97, 150, 255], dtype = "uint8")
    #Dark
    min_color = np.array([85, 40, 40], dtype = "uint8")
    max_color = np.array([100, 250, 250], dtype = "uint8")
    
    # RGB Extraction channel
    # red = frame.copy()
    # green = frame.copy()
    # blue = frame.copy()
    # red[:,:,0] = 0
    # red[:,:,1] = 0
    
    # green[:,:,0] = 0
    # green[:,:,2] = 0
    
    # blue[:,:,1] = 0
    # blue[:,:,2] = 0

    # cv2.imshow('RED Camera Vision', red)
    # cv2.imshow('GREEN Camera Vision', green)
    # cv2.imshow('BLUE Camera Vision', blue)

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    mask = cv2.inRange(HSV, min_color, max_color)
    # blur = cv2.medianBlur(mask, 5)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    # nonoise = cv2.dilate(blur, kernel)

    # filterImg = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
    filterImg = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))

    contours, hierarchy = cv2.findContours(filterImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # x_box = [640]
    # y_box = [480]
    # w_box = [0]
    # h_box = [0]
    wh_box = [0]
    coordinateFull = [(0,0,0,0)]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # print("x,y,w,h : ",x,y,w,h)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(25,100,255),2)
        wh = w*h
        wh_box.append(wh)
        coordinateFull.append((x,y,w,h))
        # x_box.append(x)
        # y_box.append(y)
        # w_box.append(w)
        # h_box.append(h)
        # if x<=x_box[0]:
        #     x_box.clear()
        #     x_box.append(x)
        # if y<=y_box[0]:
        #     y_box.clear()
        #     y_box.append(y)
        # if w>=w_box[0]:
        #     w_box.clear()
        #     w_box.append(w)
        # if h>=h_box[0]:
        #     h_box.clear()
        #     h_box.append(h)
        
        # roi=im[y:y+h,x:x+w]
        # print("X nya ",x_box)
    indexMaxBox = wh_box.index(max(wh_box))
    x,y,w,h = coordinateFull[indexMaxBox]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
    centerX = int(x+w/2)
    centerY = int(y+h/2)
    cv2.ellipse(frame,(centerX, centerY),(5,5),0,0,360,(0,0,255),2)
    cv2.ellipse(frame,(centerX, centerY),(25,25),0,0,360,(0,0,255),2)
    cropImage = filterImg[y:y+h, x:x+w]
    wh_box.clear()
    wh_box.append(0)
    coordinateFull.clear()
    coordinateFull.append((0,0,0,0))
    # cv2.rectangle(frame,(x_box[0],y_box[0]),(x_box[0]+w_box[0],y_box[0]+h_box[0]),(0,0,255),2)
    # print(x_box[0])
        # # get the min area rect
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # # convert all coordinates floating point values to int
        # box = np.int0(box)
        # print("box = ", box)
        # # draw a red 'nghien' rectangle
        # cv2.drawContours(frame, [box], 0, (0, 0, 255))

        # # finally, get the min enclosing circle
        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # # convert all values to int
        # center = (int(x), int(y))
        # radius = int(radius)
        # # and draw the circle in blue
        # cv2.circle(frame, center, radius, (255, 0, 0), 2)

    cv2.imshow('Original Camera Vision', frame)
    # cv2.imshow('HSV Camera Vision', HSV)
    cv2.imshow('MASK Camera Vision', mask)
    # cv2.imshow('BLUR Camera Vision', blur)
    # cv2.imshow('NONOISE Camera Vision', nonoise)
    cv2.imshow('FILTERIMG Camera Vision', filterImg)
    # print("width = ",cropImage.shape[0])
    # print("height = ",cropImage.shape[1])
    if (cropImage.shape[0] != 0) and (cropImage.shape[1] != 0):
        # print("show")
        resized = cv2.resize(cropImage, (20,20), interpolation = cv2.INTER_AREA) 
        cv2.imshow('CROP Camera Vision', cropImage)
        cv2.imshow('CROP RESIZED Camera Vision', resized)
    # cv2.imshow('IMG Camera Vision', img)
    # print("contours is : ",contours)
    # print("hierarchy is : ", hierarchy)
    
    # num+=1
    # print ("---------", num, "---------")
    # if num <30:
    #     pass
    # elif num>=30 and num <= 280 and num%5!=0:
    #     pass
    # elif num>=30 and num <= 280 and num%5==0:
    #     name = str('Saved'+str(num)+'.jpg')
    #     cv2.imwrite(name,frame)
    #     # name2 = str('SavedMask'+str(num)+'.jpg')
    #     # cv2.imwrite(name2,mask)
    #     name3 = str('SavedResult'+str(num)+'.jpg')
    #     cv2.imwrite(name3,filterImg)

    # else:
    #     startcam=False
    if cv2.waitKey(1) & 0xFF == ord('q'):
        startcam = False
    time.sleep(0.05)
cam.release()
cv2.destroyAllWindows()