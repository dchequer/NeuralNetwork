from abc import ABC, abstractmethod
import numpy as np

class Cost(ABC):
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    @abstractmethod
    def cost(predictedOutputs: np.ndarray, expectedOutputs: np.ndarray) -> float:
        pass
    
    @staticmethod
    @abstractmethod
    def derivative(predictedOutput: float, expectedOutput: float) -> float:
        pass

class MeanSquaredError(Cost):
    def __init__(self, name: str):
        super().__init__("MeanSquaredError")
    
    def cost(predictedOutputs: np.ndarray, expectedOutputs: np.ndarray) -> float:
        '''
        Mean Squared Error Implementation
        MSE = sum of all pairs (x, y): 0.5 * (x-y)^2
        
        '''
        cost = 0
        for predictedOutput, expectedOutput in zip(predictedOutputs, expectedOutputs):
            localError = (predictedOutput - expectedOutput)
            cost += localError * localError
        
        return 0.5 * cost
    
    def derivative(predictedOutput: float, expectedOutput: float) -> float:
        return predictedOutput - expectedOutput

class CrossEntropy(Cost):
    def __init__(self, name: str):
        super().__init__("CrossEntropy")
    
    def cost(predictedOutputs: np.ndarray, expectedOutputs: np.ndarray) -> float:
        '''
        Use when Expected Outputs are all either 0 or 1
        
        Sum over all pairs of values: yln(a) + (1 - y)ln(1 - a)

        return -1/n * Sum

        Where 
            y = corresponding expected output 
            a = corresponding predicted output
        '''
        cost = 0
        for predictedOutput, expectedOutput in zip(predictedOutputs, expectedOutputs):
            localError = expectedOutput * np.log(predictedOutput) + (1 - expectedOutput)*np.log(1 - predictedOutput)
            cost += localError
        
        return -1/len(predictedOutput) * cost
    
    def derivative(predictedOutput: float, expectedOutput: float) -> float:
        if (predictedOutput == 0 or predictedOutput == 1):
            return 0
        
        return (-predictedOutput + expectedOutput) / (predictedOutput * (predictedOutput - 1))