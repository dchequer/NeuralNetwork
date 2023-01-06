from __future__ import annotations
from Activation import *
from Cost import *
from typing import Tuple, List
from pprint import pprint

class Layer:
    def __init__(self, activation:Activation, inputSize:int = 0, outputSize:int = 0, **kwargs):
        '''
        Creates layer with random weights and bias, inherits activation function form architecture

        inputSize: m (number of neurons in current layer) | outputSize: n (number of neurons in next layer)

        '''
        self.inNodes = inputSize
        self.outNodes = outputSize

        self.bias: float = np.random.rand(outputSize) #List[n] of biases
        self.weights: np.ndarray = np.random.rand(inputSize, outputSize) #List[n] of List[m] of weights



        if kwargs:
            try:
                self.bias = kwargs['bias']
                self.weights = kwargs['weights']
                
                self.inNodes = len(self.bias)
                self.outNodes = len(self.weights)
            except KeyError:
                print('Incorrect keword arguments, only support "biases" and "weights"')

        self.costGradientWeights: np.ndarray = np.zeros(shape=self.weights.shape)
        self.costGradientBias: np.ndarray = np.zeros(shape=self.bias.shape)

        self.weightVelocities: np.ndarray = np.zeros(shape=self.weights.shape)
        self.biasVelocities: np.ndarray = np.zeros(shape=self.bias.shape) 

        self.activation: Activation = activation
        
    def __str__(self) -> str:
        return f'Layer with weights: {self.weights} and bias: {self.bias}'

    def __repr__(self) -> str:
        return self.__str__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Single Layer Forward Propagation

        '''
        outputs: np.ndarray = np.dot(inputs, self.weights) + self.bias
        #print(f'{outputs=}')
        return self.activation.activation(z=outputs)

    def forwardAndStore(self, inputs: np.ndarray) -> List[np.ndarray]:
        '''
        Single Layer Forward Propagation and Store intermediate values

        '''
        weightedInputs: np.ndarray = np.dot(inputs, self.weights) + self.bias
        activationValues = self.activation.activation(z = weightedInputs)

        return [weightedInputs, activationValues]

    def applyGradients(self, learnRate: float, regularization: float, momentum: float) -> None:
        '''
        Apply previously calculated gradients from updateGradients and reset to zero for next batch
        
        '''
        weightDecay = (1 - regularization * learnRate)

        #Deal with weights and weightVelocities
        for i, (weight, velocity) in enumerate(zip(self.weights, self.weightVelocities)):
            costGradient = self.costGradientWeights[i]

            #Calculate new weight and velocity
            newVelocity = velocity * momentum - costGradient * learnRate
            newWeight = weight * weightDecay + newVelocity

            #Update weights and velocities
            self.weightVelocities[i] = newVelocity
            self.weights[i] = newWeight

            #reset gradient
            self.costGradientWeights[i] = 0

        #Deal with biases and biasVelocities
        for i, velocity in enumerate(self.biasVelocities):
            costGradient = self.costGradientBias[i]

            #Calculate and apply new velocity and update bias
            newVelocity = velocity * momentum - costGradient * learnRate
            self.bias[i] += velocity
            
            #reset gradient
            self.costGradientBias[i] = 0
    
        return

    def updateGradients(self, inputs: np.ndarray, nodeValues: np.ndarray) -> None:
        '''
        Update gradients for connections between layers (current layer and previous)
        
        '''

        for nodeOut in range(len(nodeValues) - 1):
            for nodeIn in range(len(inputs) - 1):
                #partial derivative: cost / weight for current connection
                derivativeCostWrtWeight = inputs[nodeIn] * nodeValues[nodeOut]
                #add it to running array
                self.costGradientWeights[nodeOut][nodeIn] += derivativeCostWrtWeight
        
            #partial derivative: cost / bias for current node
            derivativeCostWrtBias = 1 * nodeValues[nodeOut]
            #add it to running array
            self.costGradientBias[nodeOut] += derivativeCostWrtBias

        return

    #nodeValues
    def partialOutputDerivatives(self, layerData: List(np.ndarray), expectedOutputs: np.ndarray, cost: Cost) -> np.ndarray:
        '''
        All partial derivatives for gradient descent contain the same last 2 partials; dC/da_2 and da_2/dz_2
        Also called costDerivative and activationDerivative, respectively
        This is a shortcut to get them

        learnData = (weightedInputs, activationValues) for each node in a layer
        expectedOutputs = [expected values] for each node in a layer

        '''

        #print(f'{layerData=}')

        weightedInputs: np.ndarray = layerData[0]
        activationValues: np.ndarray = layerData[1]

        partials: np.ndarray = np.ndarray(shape=expectedOutputs.shape) #partials are the nodeValues

        for i, expectedOutput in enumerate(expectedOutputs):
            weightedInput: float = weightedInputs[i]
            activationValue: float = activationValues[i]

            costDerivative: float = cost.derivative(predictedOutput=activationValue, expectedOutput=expectedOutput)
            activationDerivative: float = self.activation.derivative(z=weightedInput)

            partials[i] = costDerivative * activationDerivative
        
        return partials

    def partialHiddenDerivatives(self, layerData: List(np.ndarray), previousLayer: Layer, previousNodeValues: np.ndarray) -> np.ndarray:
        '''
        Calculate "nodeValues" for hidden layers
        Evaluates partial derivatives of cost wrt weighted input
        
        '''
        weightedInputs: np.ndarray = layerData[0]
        activationValues: np.ndarray = layerData[1]

        newNodeValues: np.ndarray = np.ndarray(shape=previousNodeValues.shape)
        for newNode in range(len(newNodeValues)-1):
            
            newNodeValue: float = 0
            for previousNode in range(len(previousNodeValues)-1):
                weightedInputDerivative: float = previousLayer.weights[newNode][previousNode]
                newNodeValue += weightedInputDerivative * previousNodeValues[previousNode]
            
            newNodeValue *= self.activation.derivative(weightedInputs[newNode])
            newNodeValues[newNode] = newNodeValue
    
        return newNodeValues

if __name__ == '__main__':
    inputs = np.array([0, 1])

    #print(layer.forward(inputs=inputs))
    
    customLayer = Layer(activation=Sigmoid, bias=np.array([0.54, 0.15, 0.66, 0.91]), weights=np.array([[0.7 , 0.36, 0.76, 0.04], [0.99, 0.21, 0.45, 0.03]]))
    print(f'result: {customLayer.forward(inputs=inputs)}')
