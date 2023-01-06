# NeuralNetwork
- - -

## About the Project
This is a project that I created in order to learn more about how Machine Learning works.
Might not be the most efficient compared to other options but it does work.

## Technical 
I really wanted to be able to implement various error/cost functions and different activation functions. There are a few of them available, by using
abstract methods and inheritance it is really simple and easy to read/write

### Activation
Each Activation type object must implement its own activation function along with the derivative like so,

```python
def activation(z: float, **kwargs) -> float:
  #cool stuff
  #set activation = some function
  return a
```

And a derivative,

```python
def derivative(z: float, **kwargs) -> float:
  #other cool stuff
  #set dA = some function
  return dA
 ```
 
 ### Cost
 Just like with the Activation objects, Cost objects implement the abstract method activation and derivative
 
