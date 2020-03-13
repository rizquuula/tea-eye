import numpy as np
import cv2
import time

# The duration in seconds of the video captured
# capture_duration = 600

# cap = cv2.VideoCapture(0)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('VideoTenThousandCave.avi',fourcc, 20.0, (640,480))

# start_time = time.time()
# # while( int(time.time() - start_time) < capture_duration ):
# while True:
#     ret, frame = cap.read()
#     # cv2.imshow('Original',frame)
#     if ret==True:
#         # frame = cv2.flip(frame,0)
        
#         out.write(frame)
#         cv2.imshow('frame',frame)
#     else:
#         break
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()
now = time.time()
print(now)