# -*- coding: utf-8 -*-

import sys
import math
import datetime
import enum
import numpy as np


class Function(enum.Enum):

    NONE                = 0
    THRESHOLD           = 1
    SIGMOID             = 2
    HYPERBOLIC_TANGENT  = 3


class Connection:
    
    def __init__(self):
        
        np.random.seed(datetime.datetime.now())
        self.weight = np.random.rand()
        self.deltaWeight = np.random.rand()
        

class Neuron:
    
    eta = 0.15  # overall net learning rate
    alpha = 0.5 # momentum, multiplier of last deltaWeight, [0.0..n]
    
    def __init__(self, numOutputs, index, function=Function.SIGMOID):
        
        self.weights = []*numOutputs
        for i in range(numOutputs):
            self.weights[i] = Connection()
            
        self.myIndex = index
        self.function = function
        self.output = None
        self.gradient = None
        
    def __activationFunction(self, x):
        
        result = 0.0
        if (self.function == Function.THRESHOLD):
            if (x >= 0.0):
                result = 1.0
            else:
                result = 0.0
        elif (self.function == Function.SIGMOID):
            result = 1.0/(1.0 + math.exp(-x))
        elif (self.function == Function.HYPERBOLIC_TANGENT):
            result = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        else:
            message = 'Error: unknown function!'
            sys.exit(message)
        return result

    def __activationFunctionDerivative(self, x):
        
        result = 0.0
        if (self.function == Function.THRESHOLD):
            result = 0.0
        elif (self.function == Function.SIGMOID):
            result = (1.0 - self.__activationFunction(x))*self.__activationFunction(x)
        elif (self.function == Function.HYPERBOLIC_TANGENT):
            result = 1.0 - self.__activationFunction(x)*self.__activationFunction(x)
        else:
            message = 'Error: unknown function!'
            sys.exit(message)
        return result
        
    def __sumDOW(self, nextLayer):
        
        summation = 0.0
        
        for i in range(len(nextLayer) - 1):
            summation += self.weights[i].weight*nextLayer[i].gradient;
        return summation
        
    def feedForward(self, prevLayer):
        
        summation = 0.0
        
        for i in range(len(prevLayer)):
            summation += prevLayer[i].output*prevLayer[i].weights[self.myIndex].weight
        self.output = self.__activationFunction(summation)
        
    def calcOutputGradients(self, target):
        
        delta = target - self.output
        self.gradient = delta*self.__activationFunctionDerivative(self.output)
        
    def calcHiddenGradients(self, nextLayer):
        
        dow = self.__sumDOW(nextLayer)
        self.gradient = dow*self.__activationFunctionDerivative(self.output)
    
    def updateInputWeights(self, prevLayer):
        
        # The weights to be updated are in the Connection container 
        # in the neurons in the preceding layer
        for i in range(len(prevLayer)):
            neuron = prevLayer[i]
            oldDeltaWeight = neuron.weights[self.myIndex].deltaWeight
            newDeltaWeight = Neuron.eta*neuron.output*self.gradient*Neuron.alpha*oldDeltaWeight
            
            neuron.weights[self.myIndex].deltaWeight = newDeltaWeight
            neuron.weights[self.myIndex].weight += newDeltaWeight
