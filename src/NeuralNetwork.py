from Layer import *
from Data import *
from typing import List, Dict
from alive_progress import alive_bar

import pickle

class NeuralNetwork:
    def __init__(self, layerSizes: List[int], activation:Activation, cost: Cost):
        '''
        Creates a list of layers and initializes their sizes using layerSizes, inputLayer is ignored 
        '''
        self.layers: List[Layer] = []
        previousLayerSize: int = layerSizes[0]
        for layerIndex, curLayerSize in enumerate(layerSizes[1:]):
            self.layers.append(Layer(activation=activation, inNodes=curLayerSize, previousLayerSize=previousLayerSize))
            previousLayerSize = curLayerSize

        self.cost: Cost = cost
        self.activation: Activation = activation

    def fullForward(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Repeteadly Propagate Forward

        '''
        for layer in self.layers:
            inputs = layer.forward(inputs=inputs)
        return inputs.flatten()
    
    def fullForwardAndStore(self, inputs: np.ndarray) -> Dict[int, List[np.ndarray]]:
        '''
        Repeteadly Propagate Forward 
        but also store all Layers' parameters (weightedInputs (z) and activationValues (a))
        
        '''
        NetworkData: Dict[List[np.ndarray]] = {} # {layerIndex: [arr[weightedInputs], arr[activationValues]]}
        weightedInputs = inputs
        for layerIndex, layer in enumerate(self.layers):
            weightedInputs, activationValues = layer.forwardAndStore(inputs=weightedInputs)
            NetworkData[layerIndex] = [weightedInputs, activationValues]

        return NetworkData

    def outputNodeValues(self, outputLayerData: np.ndarray, expectedOutputs: np.ndarray) -> np.ndarray[float]:
        '''
        Calculate the common output node values ((dC/da) * (da/dz))
        LayerData: [arr[z], arr[a]] 
        '''
        nodeValues: List[float] = []
        for nodeWeightedInputs, nodeActivationValues, expectedOutput in zip(*outputLayerData, expectedOutputs):
            costDerivative = self.cost.derivative(predictedOutput=nodeActivationValues, expectedOutput=expectedOutput)
            activationDerivative = self.activation.derivative(z=nodeWeightedInputs)
            nodeValues.append(costDerivative * activationDerivative)

        return np.array(nodeValues) #array containing node values for each node in output layer

    #updateAllGradients
    def fullBackward(self, inputs: np.ndarray, expectedOutputs: np.ndarray) -> None:
        '''
        Repeteadly Propagate Backward, only update the gradients, but does not apply them
        
        '''
        # all network parameters | run a learning input through network and store parameters
        NetworkData: Dict[int, List[np.ndarray]] = self.fullForwardAndStore(inputs=inputs) #{layerIndex: [arr[weightedInputs], arr[activationValues]]}
        # use output layer data to find common output node values
        outputLayerIndex = len(self.layers) - 1
        outputNodeValues: np.ndarray = self.outputNodeValues(outputLayerData=NetworkData[outputLayerIndex], expectedOutputs=expectedOutputs)
        #update gradients on output layer
        self.layers[outputLayerIndex].updateGradients(inputs=inputs, nodeValues=outputNodeValues)

        rightNodeValues = outputNodeValues
        #update gradients on hidden layers
        #print(self.layers[-2::-1])
        for iterIndex, layer in enumerate(self.layers[-2::-1]): #traverse layers backwards, skip output layer since that was already taken care of
            curLayerIndex = outputLayerIndex - (iterIndex+1)
            #print(curLayerIndex)
            layerData = NetworkData[curLayerIndex]

            #update nodeValues to contain values of layers up to the right of current layer
            rightNodeValues = layer.hiddenNodeValues(layerData=layerData, rightLayer=self.layers[curLayerIndex+1], rightNodeValues=rightNodeValues) #all node values of all layers to the right)
            layer.updateGradients(inputs=inputs, nodeValues=rightNodeValues) #use said values to update the gradients for this layer

        return

    #applyAllGradients
    def learn(self, learningBatch: DataBatch, learnRate: float, regularization: float = 1, momentum: float = 0) -> float:
        '''
        Backwards propagate with all data points in learning batch
        learnRate: how much of a cost gradient is applied | how fast NN learns
        regularization: affects a little to weightDecay | leave as 1 for no effect
        momentum: this applies to weight and bias velocities and how much they will affect | leave as 0 for no velocities

        '''

        for dataPoint in learningBatch:
            self.fullBackward(inputs=dataPoint.inputs, expectedOutputs=dataPoint.expected) #update gradients
            for layer in self.layers: #actually apply all the calculated gradients
                layer.backward(learnRate=learnRate/len(learningBatch.dataPoints), regularization=regularization, momentum=momentum)
        
        return self.calculateAvgCost(dataPoints=learningBatch) #avg cost of batch after learning

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
                if epoch % (epochs*.5) == 0: # every 10% of epochs
                    print(f'Epoch {epoch} Real-Time-Cost: {self.calculateAvgCost(dataPoints=trainingData)} Avg Cost: {runningAvg/(epoch+1)}')
        
        #print(f'final avg: {self.calculateAvgCost(dataPoints=trainingData)}')
        self.saveBrain(f'brain({curAvg})')

    def test(self, testingData: DataBatch, testSize: int) -> Dict[int, List[float]]:
        testingData: List[DataPoint] = np.random.choice(a=testingData.dataPoints, size=testSize)

        predictions: Dict = {}
        for i, dataPoint in enumerate(testingData):
            testInput, testAnswer = dataPoint.inputs, dataPoint.expected
            prediction = self.getMaxOutputNeuronIndex(testInput)
            print(f'testInput: {testInput} testAnswer: {testAnswer} prediction: {prediction[0]}')
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
    '''
    NN = NeuralNetwork(layerSizes=[1, 3, 2], activation=Sigmoid, cost=MeanSquaredError)
    
    dataPoints = [DataPoint(inputs=np.array([0]), expected=np.array([1, 0]))]

    totalCost = NN.calculateAvgCost(dataPoints=dataPoints)
    
    NN.fullBackward(dataPoint=dataPoints[0])
    '''
