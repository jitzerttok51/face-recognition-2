# Model: https://drive.google.com/u/0/uc?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1&export=download

MODEL_URL = "https://drive.google.com/u/0/uc?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1&export=download"

import urllib.request as req
import urllib.response as resp
from http.client import HTTPMessage
import os
import os.path as path
import cv2
import numpy as np
import mtcnn
import keras
import sklearn.svm
import sklearn.preprocessing
import sklearn.metrics

def getModel():
    cache = path.join('cache')
    os.makedirs(cache, exist_ok=True)
    model = path.join(cache, 'facenet_keras.h5')
    if path.exists(model):
        return model

    with req.urlopen(MODEL_URL) as u:
        meta: HTTPMessage = u.info()
        size = int(meta["Content-Length"])
        print(f"Downloading facenet_keras.h5 Bytes: {size}")

        with open(model, 'wb') as f:
            bufferSize = 8192
            total = size
            downloaded = 0

            while True: 

                buffer = u.read(bufferSize)
                if not buffer:
                    break

                f.write(buffer)
                downloaded += len(buffer)
                print(f"Downloading: {(downloaded * 1.0) / total * 100.0}%")
    return model

# from keras.models import load_model
# print(getModel())
# model = load_model('cache\\facenet_keras.h5')
# print(model.inputs)
# print(model.outputs)

detector = mtcnn.MTCNN()

def loadImage(path, size=(160,160)): 
   img = cv2.imread(path)
   result = detector.detect_faces(img)
   if not result:
       return None
   x, y, w, h = result[0]['box']
   #cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), thickness=3)
   x1 = abs(x)
   y1 = abs(y)
   x2 = x1 + w
   y2 = y1 + h

   img = img[y1:y2, x1:x2]
   img = cv2.resize(img, size)

   return img

# for pth in os.listdir(base):
#     fld = path.join(base, pth)
#     print(fld)
#     images = [loadImage(path.join(fld, f)) for f in os.listdir(fld)]
#     full = np.concatenate(images, axis=1)
#     cv2.imshow(pth,full)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def loadFromFolder(base):
    images = []
    for f in os.listdir(base):
        pth = path.join(base, f)
        img = loadImage(pth)
        if img is not None:
            images.append(img)
    return images

def loadDataset(base):
    x = []
    y = []

    for f in os.listdir(base):
        pth = path.join(base, f)
        if not path.isdir(pth):
            continue
        faces = loadFromFolder(pth)
        for face in faces:
            x.append(face)
            y.append(f)
        print(f"Loaded {len(faces)} samples for class {f}")
    return np.asarray(x), np.asarray(y)

DATASET = 'dataset/dataset.npz'
def cacheDataset():
    xTrain, yTrain = loadDataset('dataset/train')
    xTest, yTest = loadDataset('dataset/val')

    np.savez_compressed(DATASET, xTrain, yTrain, xTest, yTest)

def loadDatasets():
    if not path.exists(DATASET):
        cacheDataset()
    
    data = np.load(DATASET)
    return data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']



EMBEDDINGS_DATASET = 'dataset/embeddings-dataset.npz'

def cacheEmbeddingsDataset():
    xTrain, yTrain, xTest, yTest = loadDatasets()
    model = keras.models.load_model(getModel())
    print('Loaded Model')

    def getEmbedding(image: np.ndarray): 
        image = image.astype('float32')
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
        samples = np.expand_dims(image, axis=0)
        embedding = model.predict(samples)
        return embedding[0]

    def getEmbeddings(data: np.ndarray): 
        return np.asarray([getEmbedding(point) for point in data])

    xTrainEmbeddings = getEmbeddings(xTrain)
    print(xTrainEmbeddings.shape)

    xTestEmbeddings = getEmbeddings(xTest)
    print(xTestEmbeddings.shape)

    np.savez_compressed(EMBEDDINGS_DATASET, xTrainEmbeddings, yTrain, xTestEmbeddings, yTest)

def loadEmbeddingsDataset():
    if not path.exists(EMBEDDINGS_DATASET):
        cacheEmbeddingsDataset()
    
    data = np.load(EMBEDDINGS_DATASET)
    return data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

xTrain, yTrain, xTest, yTest = loadEmbeddingsDataset()

print(f'Dataset train: {xTrain.shape} test: {xTest.shape}')

encoderIn = sklearn.preprocessing.Normalizer(norm='l2')
xTrain = encoderIn.transform(xTrain)
xTest = encoderIn.transform(xTest)

encoderOut = sklearn.preprocessing.LabelEncoder()
encoderOut.fit(yTrain)
yTrain = encoderOut.transform(yTrain)
yTest = encoderOut.transform(yTest)

model = sklearn.svm.SVC(kernel='linear', probability=True)
model.fit(xTrain, yTrain)

trainPred = model.predict(xTrain)
testPred = model.predict(xTest)

scoreTrain = sklearn.metrics.accuracy_score(yTrain, trainPred)
scoreTest = sklearn.metrics.accuracy_score(yTest, testPred)

print('Accuracy: train=%.3f, test=%.3f' % (scoreTrain*100, scoreTest*100))