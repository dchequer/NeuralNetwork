from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    @abstractmethod
    def activation(z: float, **kwargs) -> float | np.ndarray[float]:
        pass
    
    @staticmethod
    @abstractmethod
    def derivative(z: float, **kwargs) -> float | np.ndarray[float]:
        pass

class Sigmoid(Activation):
    def __init__(self):
        super().__init__("Sigmoid")

    def activation(z, *args) -> float | np.ndarray[float]:
        return 1 / (1 + np.exp(-z))
    
    def derivative(z, *args) -> float | np.ndarray[float]:
        t = Sigmoid.activation(z)
        return t * (1 - t)

class ReLU(Activation):
    def __init__(self):
        super().__init__("ReLU")
    
    def activation(z, **kwargs) -> float:
        return np.maximum(0, z)
    
    def derivative(dA, z, **kwargs) -> float:
        dZ = np.array(dA, copy=True)
        dZ[z <= 0] = 0
        return dZ

class LeakyReLU(Activation):
    def __init__(self):
        super().__init__("LeakyReLU")
    
    def activation(z, **kwargs) -> float:
        if kwargs:
            return np.maximum(kwargs["alpha"] * z, z)
        return np.maximum(0.01 * z, z) # Default alpha

class TanH(Activation):
    def __init__(self):
        super().__init__("TanH")
    
    def activation(z, **kwargs) -> float:
        return np.tanh(z)