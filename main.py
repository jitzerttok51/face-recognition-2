import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera: cv2.VideoCapture = cv2.VideoCapture(0)

while True:

    _, img = camera.read()

    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = detector.detectMultiScale(grayScale, 1.1, 4)

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+w), (0,0,255), thickness=3)

    cv2.imshow('Camera', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

camera.release()