import cv2
import os
import os.path as path
import time

import mtcnn

def calculateAccuracy(predicate):
    def checkImage(path): 
        img = cv2.imread(path)
        return predicate(img)
    base = 'dataset/train'; checks = []; images = []
    for folder in os.listdir(base):
        src = path.join(base, folder)
        if path.isdir(src):
            images = images + [path.join(src, image) for image in os.listdir(src)]
    start = time.time()
    for i in range(len(images)):
        image: str = images[i]
        disp = image.replace("\\", "/")
        result = checkImage(image)
        print(f"({i+1}/{len(images)}) [Answer: {result}] {disp}")
        checks.append(result)
    end = time.time()
    return (sum(checks) / len(checks)) * 100, end - start


def haarMethod():
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    def check(img): 
        grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(grayScale, 1.1, 4)
        return 1 if len(boxes) > 0 else 0
    return check


def mtcnnMethod():
    detector = mtcnn.MTCNN()
    def check(img): 
        result = detector.detect_faces(img)
        return 1 if len(result) > 0 else 0
    
    return check


accuracyHaar, timeTakenHaar = calculateAccuracy(haarMethod())
accuracyMtcnn, timeTakenMtcnn = calculateAccuracy(mtcnnMethod())
print(f"Haar -> Accuraccy: {accuracyHaar:.1f}% Execution time: {timeTakenHaar:.3f} s")
print(f"Mtcnn -> Accuraccy: {accuracyMtcnn:.1f}% Execution time: {timeTakenMtcnn:.3f} s")