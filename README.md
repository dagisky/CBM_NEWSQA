# Conceptual-Base-Model
Deep neural network for high level reasioning  

## Table of Contents
1. [Project Motivation](###project-motivation)
3. [Installation](###Installation)
4. [Usage](###Usage)
5. [Licensing and Acknowledgements](###Licensing-and-Acknowledgements)



### Project Motivation
The combinatorial model over sets of high-level representation is a crucial part of  
our intelligent behavior thus, to achieve a more natural and complex system in NLU, a higher  level of representation is mandatory. The conceptual base theory has an initial premise 
that the bases of any natural language (NL) is conceptual base a representation that is not  necessarily ﬁxed or tied to a single or group of words but rather concepts and  conceptualization is regarded as the relationship between these representations. 

### Installation
The following libraries meke the dependencies for the model
* Python3
* PyTorch
* TensorFlow
* tensorboardX
* matplotlib


### Usage
The model works for 

Example:
The default program runs on multiple gpu's The main.py contains list of GPU's to run the program on.
```
python main.py # Run on multiple GPU's
```
However if you want to run on CPU use the _use_cuda_ option
```
python main.py --use_cuda false # Run on CPU
```
Note: Eventhough the model can run on cpu, it highly recomanded to use Multiple GPU to train the model. 

### Licensing and Acknowledgements

UESTC 