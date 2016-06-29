# -*- coding: utf-8 -*-

import sys
import math
import datetime
import enum
import random


class Function(enum.Enum):

    NONE                = 0
    THRESHOLD           = 1
    LINEAR              = 2
    SIGMOID             = 3
    HYPERBOLIC_TANGENT  = 4
    RELU                = 5


class Connection:
    
    def __init__(self):
        
        self.weight = 0.0
        self.deltaWeight = 0.0
        

class Neuron:
    
    eta = 0.15  # overall net learning rate [0.0..1.0]
    alpha = 0.5 # momentum, multiplier of last deltaWeight, [0.0..1.0]
    random.seed(datetime.datetime.now())    # initialize random seed
    
    def __init__(self, nLinks, index, function=Function.SIGMOID, learningRate=None):
        
        self.__weights = []
        self.__myIndex = index
        self.__function = function
        self.__output = 0.0
        self.__gradient = 0.0
        
        for i in range(nLinks):
            self.__weights.append(Connection())
            self.__weights[i].weight = random.random()
            
        if learningRate is not None:
            Neuron.eta = learningRate
            
    def __activationFunction(self, x):
        
        result = 0.0
        if (self.__function == Function.THRESHOLD):
            if (x >= 0.0):
                result = 1.0
            else:
                result = 0.0
        elif (self.__function == Function.LINEAR):
            result = x
        elif (self.__function == Function.SIGMOID):
            result = 1.0/(1.0 + math.exp(-x))
        elif (self.__function == Function.HYPERBOLIC_TANGENT):
            result = math.tanh(x)
        elif (self.__function == Function.RELU):
            if (x >= 0.0):
                result = x
            else:
                result = 0.0
        else:
            message = 'Error: unknown function!'
            sys.exit(message)
        return result

    def __activationFunctionDerivative(self, x):
        
        result = 0.0
        if (self.__function == Function.THRESHOLD):
            result = 0.0
        elif (self.__function == Function.LINEAR):
            result = 1.0
        elif (self.__function == Function.SIGMOID):
            # f'(x) = (1 - f(x))*f(x)
            result = (1.0 - x)*x    # argument x (output of the neuron) comes from activation function
        elif (self.__function == Function.HYPERBOLIC_TANGENT):
            # f'(x) = 1 - f(x)^2
            result = 1.0 - x**2 # argument x (output of the neuron) comes from activation function
        elif (self.__function == Function.RELU):
            if (x >= 0.0):
                result = 1.0
            else:
                result = 0.0
        else:
            message = 'Error: unknown function!'
            sys.exit(message)
        return result
        
    # Calculate summation of DOW(Descent of Weight)
    def __sumDOW(self, nextLayer):
        
        summ = 0.0
        
        for i in range(len(nextLayer) - 1): # exclude bias neuron
            summ += self.__weights[i].weight*nextLayer[i].__gradient;
        return summ
        
    def feedForward(self, prevLayer):
        
        summ = 0.0
        
        for i in range(len(prevLayer)):
            summ += prevLayer[i].__output*prevLayer[i].__weights[self.__myIndex].weight
        self.__output = self.__activationFunction(summ)
        
    # Calculate gradients of output layer
    def calcOutputGradients(self, target):
        
        delta = -(target - self.__output)
        self.__gradient = delta*self.__activationFunctionDerivative(self.__output)
        
    # Calculate gradients of each layer
    def calcHiddenGradients(self, nextLayer):
        
        dow = self.__sumDOW(nextLayer)
        self.__gradient = dow*self.__activationFunctionDerivative(self.__output)
        
    def updateWeights(self, prevLayer):
        
        # The weights to be updated are in the Connection container 
        # in the neurons in the preceding layer
        for i in range(len(prevLayer)):
            neuron = prevLayer[i]
            oldDeltaWeight = neuron.__weights[self.__myIndex].deltaWeight
            # The second term is a momentum to accelerate learning
            newDeltaWeight = -Neuron.eta*neuron.__output*self.__gradient + Neuron.alpha*oldDeltaWeight
            
            neuron.__weights[self.__myIndex].deltaWeight = newDeltaWeight
            neuron.__weights[self.__myIndex].weight += newDeltaWeight
            
    def getOutput(self):
        
        return self.__output
        
    def getWeights(self):
        
        return self.__weights
        
    def setOutput(self, y):
        
        self.__output = y
        
    def setWeights(self, weights):
        
        for i in range(len(weights)):
            self.__weights[i].weight = weights[i]
