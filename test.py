import cv2
import time

cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    now = time.time()
    delay = now-start_time
    print('fps = ', 1/delay)

cap.release()
cv2.destroyAllWindows()
