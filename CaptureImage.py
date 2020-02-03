import numpy as np
import cv2

cap = cv2.VideoCapture(0)
name = 0
while True:
    ret, frame = cap.read()

    cv2.imshow("Frame",frame)
    name+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(33) == ord('s'):
        nameFile = str('ImageTest'+str(name)+'.jpg')
        cv2.imwrite(nameFile,frame)
        print("Image saved as", name)

cap.release()
cv2.destroyAllWindows()


