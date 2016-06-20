# -*- coding: utf-8 -*-

import sys
import math
import neuron


class Network:
    
    recentAverageSmoothingFactor = 0.1
    
    def __init__(self, topology):
        
        self.layers = []
        nLayers = len(topology)
        for layerNum in range(nLayers):
            self.layers.append([])
            nNeurons = topology[layerNum]
            for i in range(nNeurons):
                if layerNum == len(topology) - 1:
                    self.layers[layerNum].append(neuron.Neuron(1,i,neuron.Function.SIGMOID))
                else:
                    self.layers[layerNum].append(neuron.Neuron(topology[layerNum+1],i,neuron.Function.SIGMOID))
                    
        self.error = 0.0
        self.recentAverageError = 0.0
    
    def feedForward(self, inputVector):
        
        if len(inputVector) != len(self.layers[0]):
            message = 'Error: dimension of input vector is not match!'
            sys.exit(message)
            
        # Assign the input values into the input layer
        for i in range(len(inputVector)):
            self.layers[0][i].output = inputVector[i]
            
        # Forward propagation
        nLayers = len(self.layers)
        for layerNum in range(1, nLayers):
            prevLayer = self.layers[layerNum-1]
            nNeurons = len(self.layers[layerNum])
            for i in range(nNeurons):
                self.layers[layerNum][i].feedForward(prevLayer)
                
    def backPropagation(self, targetVector):
        
        # RMSE(Root Mean Square Error)
        self.error = 0.0
        
        # Calculate overall net error (RMS of output neuron errors)
        outputLayer = self.layers[-1]
        for i in range(len(outputLayer)):
            delta = targetVector[i] - outputLayer[i].output
            self.error += delta**2
        self.error /= len(outputLayer)
        self.error = math.sqrt(self.error)
        
        # Implement a recent average measurement
        self.recentAverageError = (self.recentAverageError*Network.recentAverageSmoothingFactor + self.error)/(Network.recentAverageSmoothingFactor + 1.0)
        
        # Calculate output layer gradients
        for i in range(len(outputLayer)):
            outputLayer[i].calcOutputGradients(targetVector[i])
            outputLayer[i].weights[0].deltaWeight = -(targetVector[i] - outputLayer[i].output)

        # Calculate gradients on hidden layers
        nLayers = len(self.layers)
        for layerNum in range(nLayers-2, -1, -1):
            currLayer = self.layers[layerNum]
            nextLayer = self.layers[layerNum + 1]

            nNeurons = len(currLayer)
            for i in range(nNeurons):
                currLayer[i].calcHiddenGradients(nextLayer)

        # For all layers from outputs to first hidden layer, update connection weights
        for layerNum in range(nLayers-1, 0, -1):
            currLayer = self.layers[layerNum]
            prevLayer = self.layers[layerNum - 1]
            
            nNeurons = len(currLayer)
            for i in range(nNeurons):
                currLayer[i].updateWeights(prevLayer)
    
    def getResults(self):
        
        results = []
        for i in range(len(self.layers[-1])):
            results.append(self.layers[-1][i].output)
        return results
        
    def getRecentAverageError(self):
        
        return self.recentAverageError
