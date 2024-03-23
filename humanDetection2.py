import numpy as np
import cv2


bodydetection = cv2.CascadeClassifier('cascads/haarcascade_fullbody.xml')
ubodydetection = cv2.CascadeClassifier('cascads/haarcascade_lowerbody.xml')
lbodydetection = cv2.CascadeClassifier('cascads/haarcascade_upperbody.xml')


# Параметры
BODY_SCALE = 1.2
UPPER_SCALE = 1.1


image_count = 0
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    body=bodydetection.detectMultiScale(gray, BODY_SCALE, 1)
    if len(body) > 0:
        upper=ubodydetection.detectMultiScale(gray, UPPER_SCALE, 1)
        if len(upper) > 0:
            for (a,b,c,d) in upper:
                cv2.rectangle(frame, (a,b), (a+c,b+d), (0,0,255), 2)
                image_count += 1
                cv2.imwrite(f"foundImages/found{image_count}.png", frame)
                print("accept")

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
