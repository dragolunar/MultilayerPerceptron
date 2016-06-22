# -*- coding: utf-8 -*-

import sys
import math
import datetime
import enum
import random


class Function(enum.Enum):

    NONE                = 0
    THRESHOLD           = 1
    SIGMOID             = 2
    HYPERBOLIC_TANGENT  = 3
    RELU                = 4


class Connection:
    
    def __init__(self):
        
        self.weight = 0.0
        self.deltaWeight = 0.0
        

class Neuron:
    
    eta = 0.15  # overall net learning rate [0.0..1.0]
    alpha = 0.5 # momentum, multiplier of last deltaWeight, [0.0..1.0]
    random.seed(datetime.datetime.now())
    
    def __init__(self, nLinks, index, function=Function.SIGMOID, learningRate=None):
        
        self.weights = []
        for i in range(nLinks):
            self.weights.append(Connection())
            self.weights[i].weight = random.random()
            
        self.myIndex = index
        self.function = function
        self.output = 0.0
        self.gradient = 0.0
        
        if learningRate is not None:
            Neuron.eta = learningRate
        
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
            result = math.tanh(x)
        elif (self.function == Function.RELU):
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
        if (self.function == Function.THRESHOLD):
            result = 0.0
        elif (self.function == Function.SIGMOID):
            # f'(x) = (1 - f(x))*f(x)
            result = (1.0 - self.__activationFunction(x))*self.__activationFunction(x)
        elif (self.function == Function.HYPERBOLIC_TANGENT):
            # f'(x) = 1 - f(x)^2
            result = 1.0 - math.tanh(x)**2
        elif (self.function == Function.RELU):
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
            summ += self.weights[i].weight*nextLayer[i].gradient;
        return summ
        
    def feedForward(self, prevLayer):
        
        summ = 0.0
        
        for i in range(len(prevLayer)):
            summ += prevLayer[i].output*prevLayer[i].weights[self.myIndex].weight
        self.output = self.__activationFunction(summ)
        
    # Calculate gradients of output layer
    def calcOutputGradients(self, target):
        
        delta = -(target - self.output)
        self.gradient = delta*self.__activationFunctionDerivative(self.output)
        
    # Calculate gradients of each layer
    def calcHiddenGradients(self, nextLayer):
        
        dow = self.__sumDOW(nextLayer)
        self.gradient = dow*self.__activationFunctionDerivative(self.output)
    
    
    def updateWeights(self, prevLayer):
        
        # The weights to be updated are in the Connection container 
        # in the neurons in the preceding layer
        for i in range(len(prevLayer)):
            neuron = prevLayer[i]
            oldDeltaWeight = neuron.weights[self.myIndex].deltaWeight
            # The second term is a momentum to accelerate learning
            newDeltaWeight = -Neuron.eta*neuron.output*self.gradient + Neuron.alpha*oldDeltaWeight
            
            neuron.weights[self.myIndex].deltaWeight = newDeltaWeight
            neuron.weights[self.myIndex].weight += newDeltaWeight
            
    def setOutput(self, y):
        
        self.output = y
            
    def getOutput(self):
        
        return self.output

    def getWeights(self):
        
        return self.weights
