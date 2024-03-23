import numpy as np
import cv2

cascade = cv2.CascadeClassifier('cascads/haarcascade_upperbody.xml');

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        body = cascade.detectMultiScale(
            gray,
            scaleFactor = 1.02,
            minNeighbors = 3,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        temp = 0
        if body[0][0]:
            for (x, y, w, h) in body:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Upper Body', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
