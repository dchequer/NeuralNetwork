from math import isnan
from Layer import *
from Data import *
from typing import List, Dict
from alive_progress import alive_bar

import pickle

class NeuralNetwork:
    def __init__(self, layerSizes: List[int], activation:Activation, cost: Cost):
        self.layers: List[Layer] = []
        for i, size in enumerate(layerSizes):
            try: nextSize = layerSizes[i+1] 
            except IndexError: nextSize = size

            self.layers.append(Layer(activation=activation, inputSize=size, outputSize=nextSize))

        self.cost: Cost = cost

    def fullForward(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Repeteadly Propagate Forward
        '''
        for i, layer in enumerate(self.layers):
            inputs = layer.forward(inputs=inputs)
        return inputs
    
    def fullBackward(self, dataPoint: DataPoint) -> None:
        '''
        Repeteadly Propagate Backward
        '''
        # collection of all attributes for each layer and each node in said layer (weightedInputs, activationValues) 
        learnData: Dict[List[np.ndarray]] = {} # {layerIndex: [arr[weightedInputs], arr[activationValues]]}
        weightedInputs = dataPoint.inputs
        for layerIndex, layer in enumerate(self.layers):
            weightedInputs, activationValues = layer.forwardAndStore(inputs=weightedInputs)
            learnData[layerIndex] = [weightedInputs, activationValues]
        
        outputLayerIndex = len(self.layers) - 1
        outputLayerLearnData: List[np.ndarray] = learnData[outputLayerIndex] #stored data from output layer after propagation

        #calculate partial derivatives for output layer and update output layer gradients
        partialOutputDerivatives = self.layers[outputLayerIndex].partialOutputDerivatives(layerData=outputLayerLearnData, expectedOutputs=dataPoint.expected, cost=self.cost) #nodeValues
        #learnData[outputLayerIndex].append(partialOutputDerivatives)
        self.layers[outputLayerIndex].updateGradients(inputs=dataPoint.inputs, nodeValues=partialOutputDerivatives)
        
        #calculate partial derivatives for hidden layers and update all hidden layer gradients
        for i in range(len(learnData)-1):
            hiddenLayerIndex = outputLayerIndex - i
            hiddenLayer = self.layers[hiddenLayerIndex]
            previousLayer = self.layers[hiddenLayerIndex - 1]

            previousNodeValues = hiddenLayer.partialOutputDerivatives(layerData=learnData[hiddenLayerIndex], expectedOutputs=dataPoint.expected, cost=self.cost)
            #learnData[hiddenLayerIndex-1].append(previousLayerNodeValues)

            partialHiddenDerivatives = hiddenLayer.partialHiddenDerivatives(layerData=learnData[hiddenLayerIndex], previousLayer=previousLayer, previousNodeValues=previousNodeValues)

            hiddenLayer.updateGradients(inputs=dataPoint.inputs, nodeValues=partialHiddenDerivatives)
        
        return

    def learn(self, learningBatch: DataBatch, learnRate: float, regularization: float = 0, momentum: float = 0) -> float:
        # Use back-propagation algorithm (fullBackward), calculate gradient of Cost function
        # WRT to weights and biases
        # dataBatch is equivalent to trainingData

        for dataPoint in learningBatch:
            self.fullBackward(dataPoint=dataPoint) #update gradients
        

        for layer in self.layers:
            layer.applyGradients(learnRate=learnRate / len(learningBatch), regularization=regularization, momentum=momentum)
        
        return self.calculateAvgCost(dataPoints=learningBatch)

    def train(self, trainingData: DataBatch, batchSize: int, epochs: int, learnRate: float, regularization: float = 0, momentum: float = 0, cool: bool = False) -> None:
        if cool:
            with alive_bar(epochs, dual_line=True, title='Learning') as bar:
                runningAvg = 0
                for epoch in range(epochs):
                    curAvg = self.learn(learningBatch=trainingData.miniBatch(size=batchSize), learnRate=learnRate, regularization=regularization, momentum=momentum)
                    runningAvg += curAvg
                    bar.text = f'Epoch {epoch} Real-Time-Cost: {curAvg} Avg-Cost: {runningAvg/(epoch+1)}'
                    bar()
        else:
            runningAvg = 0
            for epoch in range(epochs):
                curAvg = self.learn(learningBatch=trainingData.miniBatch(size=batchSize), learnRate=learnRate, regularization=regularization, momentum=momentum)
                runningAvg += curAvg
                if epoch % 250 == 0:
                    print(f'Epoch {epoch} Real-Time-Cost: {self.calculateAvgCost(dataPoints=trainingData)} Avg Cost: {runningAvg/(epoch+1)}')
        
        self.saveBrain(f'brain({curAvg})')

    def test(self, testingData: DataBatch, testSize: int) -> Dict[int, List[float]]:
        testingData: List[DataPoint] = np.random.choice(a=testingData.dataPoints, size=testSize)

        predictions: Dict = {}
        for i, dataPoint in enumerate(testingData):
            testInput, testAnswer = dataPoint.inputs, dataPoint.expected
            prediction = self.getMaxOutputNeuronIndex(testInput)
            predictions[i] = [testAnswer, prediction[0]]

        return predictions

    def calculateCost(self, dataPoint: DataPoint) -> float:
        expectedOutputs: np.ndarray = dataPoint.expected #outputs from given DataPoint (true values)
        predictedOutputs: np.ndarray = self.fullForward(inputs=dataPoint.inputs) #real outputs from NN
        
        return self.cost.cost(predictedOutputs=predictedOutputs, expectedOutputs=expectedOutputs)
    
    def calculateAvgCost(self, dataPoints: DataBatch) -> float:
        totalCost = 0
        for dataPoint in dataPoints:
            totalCost += self.calculateCost(dataPoint=dataPoint)
        return totalCost

    def saveBrain(self, path:str) -> None:
        with open(path, 'wb') as brainFile:
            pickle.dump(self, file=brainFile)
        return 
    
    def getMaxOutputNeuronIndex(self, inputs: np.ndarray) -> Dict[int, float]:
        outputs: np.ndarray = self.fullForward(inputs=inputs)
        predictions = sorted({outputIndex : outputConfidence for outputIndex, outputConfidence in enumerate(outputs)})
        return predictions


if __name__ == '__main__':
    NN = NeuralNetwork(layerSizes=[1, 3, 2], activation=Sigmoid, cost=MeanSquaredError)
    
    dataPoints = [DataPoint(inputs=np.array([0]), expected=np.array([1, 0]))]

    totalCost = NN.calculateAvgCost(dataPoints=dataPoints)
    
    NN.fullBackward(dataPoint=dataPoints[0])