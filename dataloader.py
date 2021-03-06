# -*- coding: utf-8 -*-

import sys
import re
import random
import datetime

# Silly class to read training/test data from a file
class InputData:
    
    random.seed(datetime.datetime.now())    # initialize random seed
    
    def __init__(self, fileName):
        
        self.__topology = []
        self.__inputs = []
        self.__targets = []
        self.__pos = 0
        
        fp = open(fileName, 'r')
        if fp is None:
            message = 'Error: ' + fileName + 'is not found!'
            sys.exit(message)

        line = fp.readline()
        line = line.replace('\n', '')
        if re.match('topology: ', line) is None:
            message = 'Error: file type mismatch!'
            sys.exit(message)
        line = re.sub('topology: ', '', line)
        for i in line.split(' '):
            if re.match('[0-9]+', i):
                self.__topology.append(int(i))
        
        lines = fp.readlines()
        fp.close()
        
        # Read input vector and target vector
        for line in lines:
            line.replace('\n', '')
            temp = []
            if re.match('in: ', line):
                line = re.sub('in: ', '', line)
                for i in line.split(' '):
                    if re.match('[0-9]+', i):
                        temp.append(float(i))
                self.__inputs.append(temp)
            elif re.match('out: ', line):
                line = re.sub('out: ', '', line)
                for i in line.split(' '):
                    if re.match('[0-9]+', i):
                        temp.append(float(i))
                self.__targets.append(temp)
            else:
                message = 'Error: invalid line'
                sys.exit(message)
                
    def isEof(self):
        
        if len(self.__inputs) == self.__pos:
            return True
        else:
            return False
            
    def getTopology(self):
        
        return self.__topology
        
    def getSampleSize(self):
        
        return len(self.__inputs)
        
    def getSamples(self):
        
        return self.__inputs, self.__targets
        
    def getNextValues(self):
        
        inputs = self.__inputs[self.__pos]
        targets = self.__targets[self.__pos]
        self.__pos += 1
        
        return inputs, targets
        
    def head(self):
        
        self.__pos = 0
        
    def shuffle(self):
        
        indices = list(range(len(self.__inputs)))
        random.shuffle(indices)
        
        temp_inputs = []
        temp_targets = []
        for index in indices:
            temp_inputs.append(self.__inputs[index])
            temp_targets.append(self.__targets[index])
        self.__inputs = temp_inputs
        self.__targets = temp_targets


class WeightData:
    
    def __init__(self, fileName):
        
        self.__topology = []
        self.__weights = []
        
        fp = open(fileName, 'r')
        if fp is None:
            message = 'Error: ' + fileName + 'is not found!'
            sys.exit(message)
            
        line = fp.readline()
        line = line.replace('\n', '')
        if re.match('topology: ', line) is None:
            message = 'Error: file type mismatch!'
            sys.exit(message)
        line = re.sub('topology: ', '', line)
        for i in line.split(' '):
            if re.match('[0-9]+', i):
                self.__topology.append(int(i))
        
        lines = fp.readlines()
        fp.close()
        
        # Weights list initialization
        nLayers = len(self.__topology) - 1  # ignore output layer
        for layerNum in range(nLayers):
            self.__weights.append([])
            nNeurons = self.__topology[layerNum] + 1   # add a bias neuron
            for i in range(nNeurons):
                self.__weights[layerNum].append([])
        
        for line in lines:
            line.replace('\n', '')
            
            vals = line.split(' ')
            layerNum = int(vals[0])
            k = int(vals[1])
            j = int(vals[2])
            weight = float(vals[3])
            
            if self.__weights[layerNum][k] is None:
                self.__weights[layerNum][k] = []
            self.__weights[layerNum][k].append(weight)
            
    def getTopology(self):
        
        return self.__topology
        
    def getWeightValues(self):
        
        return self.__weights
