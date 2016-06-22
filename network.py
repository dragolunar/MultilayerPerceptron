# -*- coding: utf-8 -*-

import sys
import math
import neuron


class Network:
    
    recentAverageSmoothingFactor = 1000.0    # Number of training samples to average over
    
    def __init__(self, topology, learningRate=None):
        
        self.layers = []
        nLayers = len(topology)
        for layerNum in range(nLayers):
            self.layers.append([])
            nNeurons = topology[layerNum] + 1   # add a bias neuron
            for i in range(nNeurons):
                if layerNum == len(topology) - 1:
                    # There is no forward link at the last layer
                    self.layers[layerNum].append(neuron.Neuron(0,i,neuron.Function.RELU,learningRate))
                else:
                    self.layers[layerNum].append(neuron.Neuron(topology[layerNum+1],i,neuron.Function.HYPERBOLIC_TANGENT,learningRate))
            # Force the bias node's output to 1.0 (it was the last neuron pushed in this layer)
            self.layers[layerNum][-1].setOutput(1.0)
                    
        self.error = 0.0
        self.recentAverageError = 0.0
    
    def feedForward(self, inputVector):
        
        if len(inputVector) != len(self.layers[0]) - 1: # exclude bias neuron
            message = 'Error: dimension of input vector is not match!'
            sys.exit(message)
            
        # Assign the input values into the input layer
        for i in range(len(inputVector)):
            self.layers[0][i].setOutput(inputVector[i])
            
        # Forward propagation
        nLayers = len(self.layers)
        for layerNum in range(1, nLayers):  # exclude input layer
            prevLayer = self.layers[layerNum-1]
            nNeurons = len(self.layers[layerNum])
            for i in range(nNeurons - 1):   # exclude bias neuron (it's output is always 1.0)
                self.layers[layerNum][i].feedForward(prevLayer)
                
    def backPropagation(self, targetVector):
        
        ERROR_TYPE = 'SE'     # SE(Square Error) or RMSE(Root Mean Square Error)
        self.error = 0.0
        
        # Calculate overall net error (SE of output neuron errors)
        outputLayer = self.layers[-1]
        for i in range(len(outputLayer) - 1):   # exclude bias neuron
            delta = targetVector[i] - outputLayer[i].getOutput()
            self.error += delta**2
            
        if ERROR_TYPE == 'SE':
            self.error /= 2.0
        elif ERROR_TYPE == 'RMSE':
            self.error /= (len(outputLayer) - 1)    # get average error squared
            self.error = math.sqrt(self.error)      # RMS
        
        # Implement a recent average measurement
        self.recentAverageError = (self.recentAverageError*Network.recentAverageSmoothingFactor + self.error)/(Network.recentAverageSmoothingFactor + 1.0)
        
        # Calculate output layer gradients
        for i in range(len(outputLayer) - 1):   # exclude bias neuron
            outputLayer[i].calcOutputGradients(targetVector[i])

        # Calculate gradients on hidden layers
        nLayers = len(self.layers)
        for layerNum in range(nLayers-2, 0, -1):
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
            for i in range(nNeurons - 1):   # exclude bias neuron
                currLayer[i].updateWeights(prevLayer)
    
    def getResults(self):
        
        results = []
        for i in range(len(self.layers[-1]) - 1):   # exclude bias neuron
            results.append(self.layers[-1][i].getOutput())
        return results
        
    def getRecentError(self):
        
        return self.error
        
    def getRecentAverageError(self):
        
        return self.recentAverageError
