from __future__ import annotations
from Activation import *
from Cost import *
from typing import List


class Layer:
    def __init__(
        self, activation: Activation, inNodes: int, previousLayerSize: int, **kwargs
    ):
        """
        Creates layer with random weights and bias, inherits activation function from Activation and cost from Cost

        inNodes: m (number of neurons in current layer) | previousLayerSize: n (number of neurons in previous layer)

        """
        self.inNodes = inNodes
        # self.outNodes = outNodes

        self.bias: float = np.random.rand(
            inNodes, 1
        )  # List[m] of biases | one bias for each neuron in current layer
        self.weights: np.ndarray = np.random.rand(
            inNodes, previousLayerSize
        )  # m Lists of List[n] of weights | one weight for each neuron in previous layer, for each node in current layer

        if kwargs:
            try:
                self.bias = kwargs["bias"]
                self.weights = kwargs["weights"]

                self.inNodes = len(self.bias)
                # self.outNodes = len(self.weights)
            except KeyError:
                print('Incorrect keword arguments, only support "biases" and "weights"')

        self.costGradientWeights: np.ndarray = np.zeros(shape=self.weights.shape)
        self.costGradientBias: np.ndarray = np.zeros(shape=self.bias.shape)

        self.weightVelocities: np.ndarray = np.zeros(shape=self.weights.shape)
        self.biasVelocities: np.ndarray = np.zeros(shape=self.bias.shape)

        self.activation: Activation = activation

    def __str__(self) -> str:
        return f"Layer with weights: {self.weights} and bias: {self.bias}"

    def __repr__(self) -> str:
        return self.__str__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Single Layer Forward Propagation

        """
        outputs: np.ndarray = np.dot(self.weights, inputs) + self.bias
        return self.activation.activation(outputs)

    def forwardAndStore(self, inputs: np.ndarray) -> List[np.ndarray]:
        """
        Single Layer Forward Propagation and Store intermediate values

        """
        weightedInputs: np.ndarray = np.dot(self.weights, inputs) + self.bias
        activationValues = self.activation.activation(z=weightedInputs)

        return [weightedInputs, activationValues]

    # apply gradients
    def backward(
        self, learnRate: float, regularization: float, momentum: float
    ) -> None:
        """
        Apply previously calculated gradients from updateGradients and reset to zero for next batch

        """
        # determine decay rate
        weightDecay = 1 - regularization * learnRate
        # print(f'weightDecay: {weightDecay}')
        # Deal with weights and weightVelocities
        for i, (weight, velocity) in enumerate(
            zip(self.weights, self.weightVelocities)
        ):
            costGradient = self.costGradientWeights[i]

            # Calculate new weight and velocity
            newVelocity = velocity * momentum - costGradient * learnRate
            newWeight = weight * weightDecay + newVelocity

            # Update weights and velocities
            self.weightVelocities[i] = newVelocity
            self.weights[i] = newWeight

            # reset gradient
            # print(f'''
            # costGradient: {costGradient}
            # Velocity (old vs new): {velocity} -> {newVelocity}
            # newWeight (old vs new): {weight} -> {newWeight}
            #''')
            self.costGradientWeights[i] = 0

        # Deal with biases and biasVelocities
        for i, velocity in enumerate(self.biasVelocities):
            costGradient = self.costGradientBias[i]

            # Calculate and apply new velocity and update bias
            newVelocity = velocity * momentum - costGradient * learnRate
            self.bias[i] += velocity

            # reset gradient
            self.costGradientBias[i] = 0
        return

    def updateGradients(self, inputs: np.ndarray, nodeValues: np.ndarray) -> None:
        """
        Update gradients for connections between layers (current layer and previous)
        For weights:
            dC/dw = nodeValues * input

        For bias:
            dC/db = nodeValues

        """
        for nodeToIdx, nodeValue in enumerate(nodeValues):
            for nodeFromIdx, nodeInput in enumerate(inputs):
                derivativeCostWRTWright = nodeValue * nodeInput

                # update weight gradients array
                self.costGradientWeights[nodeToIdx][
                    nodeFromIdx
                ] += derivativeCostWRTWright

            derivativeCostWRTBias = nodeValue

            # update bias gradients array
            self.costGradientBias[nodeToIdx] += derivativeCostWRTBias

        return

    def hiddenNodeValues(
        self,
        layerData: List[np.ndarray],
        rightLayer: Layer,
        rightNodeValues: np.ndarray,
    ) -> np.ndarray:
        """
        previousNodeValues are originally just the output nodeValues
        However with each layer, 2 new nodeValues are multiplied to that

        LayerData: [arr[z], arr[a]]
        """
        weightedInputs = layerData[0]

        leftNodeValues: np.ndarray = np.zeros(shape=rightNodeValues.shape)
        for leftIdx, leftNodeValue in enumerate(leftNodeValues):
            for rightIdx, rightNodeValue in enumerate(rightNodeValues):
                weightedInputDerivative = rightLayer.weights[leftIdx][
                    rightIdx
                ]  # dz/da = w[from][to]
                leftNodeValue += weightedInputDerivative * rightNodeValue

            leftNodeValue *= self.activation.derivative(weightedInputs[leftIdx])
            leftNodeValues[leftIdx] = leftNodeValue

        return leftNodeValues


if __name__ == "__main__":
    inputs = np.array([0, 1])

    # print(layer.forward(inputs=inputs))

    customLayer = Layer(
        activation=Sigmoid,
        bias=np.array([0.54, 0.15, 0.66, 0.91]),
        weights=np.array([[0.7, 0.36, 0.76, 0.04], [0.99, 0.21, 0.45, 0.03]]),
    )
    print(f"result: {customLayer.forward(inputs=inputs)}")
