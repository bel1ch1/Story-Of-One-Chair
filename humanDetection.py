import cv2
import time

person_cascade = cv2.CascadeClassifier('humanCascad/haarcascade_fullbody.xml')
#cap = cv2.VideoCapture('People .mkv') # for video
cap = cv2.VideoCapture(0) # webcam video

img_index = 0
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame,(640,360))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Haar-cascade classifier needs a grayscale image
        rects = person_cascade.detectMultiScale(gray_frame)
        
            
        # for (x, y, w, h) in rects:
        #     cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.imshow("preview", frame)
        #cv2.imwrite('img/img'+str(img_index)+'.jpg',frame)
        img_index += 1
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break
