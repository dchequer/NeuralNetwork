from __future__ import annotations
from NeuralNetwork import *
import matplotlib.pyplot as plt
from gzip import GzipFile, open

def createTrainingBatch(batchSize: int, inputSize: int, outputSize: int) -> DataBatch:
    inputsArr: np.ndarray = []
    expectedArr: np.ndarray = []
    for _ in range(batchSize):
        inputs: np.ndarray = np.random.rand(inputSize)
        expected: np.ndarray = np.random.rand(outputSize)

        inputsArr.append(inputs)
        expectedArr.append(expected)
    
    trainingBatch: DataBatch = DataBatch(inputsArr=inputsArr, expectedArr=expectedArr)
    
    return trainingBatch

def fileToArray(zipfile: GzipFile, imageSize:int, numImages: int) -> np.ndarray:
    buf = zipfile.read(imageSize*imageSize*numImages)
    imageArr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    imageArr = imageArr.reshape(numImages, imageSize, imageSize, 1)
    return imageArr

def showImage(imageArr: np.ndarray, imageIndex: int) -> None:
    plt.imshow(np.asarray(imageArr[imageIndex]).squeeze(), cmap=plt.cm.binary)
    plt.show()

if __name__ == '__main__':
    layerSizes = [28*28, 986, 524, 10]
    myNN = NeuralNetwork(layerSizes=layerSizes, activation=Sigmoid, cost=MeanSquaredError)

    imageSize = 28
    #open training images and labels
    trainingImagesFile = open('data/train-images-idx3-ubyte.gz')
    trainingLabelsFile = open('data/train-labels-idx1-ubyte.gz')
    trainingSize = 60_000 #total training pairs

    #open testing images and labels
    testingImagesFile = open('data/t10k-images-idx3-ubyte.gz')
    testingLabelsFile = open('data/t10k-labels-idx1-ubyte.gz')
    testingSize = 10_000 #total testing pairs


    desiredTrainingDatasetSize = 15_000
    desiredTestingDatasetSize = 2_500
    #read image files into np arrays
    trainingImagesFile.read(16) #ignore first 16 bytes, due to format
    trainingImagesArr = fileToArray(zipfile=trainingImagesFile, imageSize=imageSize, numImages=desiredTrainingDatasetSize)
    testingImagesFile.read(16)
    testingImagesArr = fileToArray(zipfile=testingImagesFile, imageSize=imageSize, numImages=desiredTestingDatasetSize)

    #read label files into np arrays
    trainingLabelsFile.read(8)
    trainingLabelsArr = fileToArray(zipfile=trainingLabelsFile, imageSize=1, numImages=desiredTrainingDatasetSize)
    testingLabelsFile.read(8)
    testingLabelsArr = fileToArray(zipfile=testingLabelsFile, imageSize=1, numImages= desiredTestingDatasetSize)

    #clean up inputs to match format
    trainingImagesArr = trainingImagesArr.reshape((desiredTrainingDatasetSize, imageSize*imageSize)) #flatten each image to be single list of 784 items
    testingImagesArr = testingImagesArr.reshape((desiredTestingDatasetSize, imageSize*imageSize))

    trainingLabelsArr = trainingLabelsArr.reshape((desiredTrainingDatasetSize, 1))
    testingLabelsArr = testingLabelsArr.reshape((desiredTestingDatasetSize, 1))
    #create training and testing data batches
    training = DataBatch(inputsArr=trainingImagesArr, expectedArr=trainingLabelsArr)
    myNN.train(trainingData=training, batchSize = -1, epochs=1000, learnRate=0.1, regularization=0, momentum=0)
    
    #show images
    #showImage(trainingImagesArr, 1)

    
