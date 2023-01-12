from __future__ import annotations
import numpy as np
from typing import List

class DataPoint:
    def __init__(self, inputs: np.ndarray, expected: np.ndarray) -> None:
        self.inputs = inputs #for MNIST these are the input pixels
        self.expected = expected #for MNIST there are the label predictions

    def __str__(self) -> str:
        return f'DataPoint with inputs: {self.inputs} and expected: {self.expected}'

    def __repr__(self) -> str:
        return self.__str__()
    
class DataBatch:
    def __init__(self, inputsArr: List[np.ndarray], expectedArr: List[np.ndarray]) -> None:
        self.curIndex = 0
        self.dataPoints: List[DataPoint] = []

        for inputs, expected in zip (inputsArr, expectedArr):
            self.dataPoints.append(DataPoint(inputs=inputs, expected=expected))

    def __str__(self) -> str:
        return f'DataBatch with {len(self.dataPoints)} data points'

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.dataPoints)

    def __iter__(self):
        return self

    def __next__(self) -> DataPoint:
        if self.curIndex >= len(self.dataPoints):
            self.curIndex = 0
            raise StopIteration
        else:
            self.curIndex += 1
            return self.dataPoints[self.curIndex - 1]
    
    def __getitem__(self, index: int) -> DataPoint:
        return self.dataPoints[index]
    
    def miniBatch(self, size: int = -1) -> DataBatch:
        '''
        Returns a random DataBatch of size 'size' from the current DataBatch
        '''
        if size > len(self.dataPoints):
            raise ValueError(f'Size of miniBatch ({size}) is larger than size of current DataBatch ({len(self.dataPoints)})')
        elif size == 0:
            raise ValueError(f'Size of miniBatch ({size}) cannot be 0')
        elif size < 0:
            size = np.random.randint(1, len(self.dataPoints)/100)
        #else:
        indices = np.random.choice(len(self.dataPoints), size=size)
        return DataBatch([self.dataPoints[index].inputs for index in indices], [self.dataPoints[index].expected for index in indices])