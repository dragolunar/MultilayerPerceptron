# -*- coding: utf-8 -*-

import sys
import re

# Silly class to read training/test data from a file
class dataloader:
    
    def __init__(self, fileName):
        
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
        self.__topology = [int(i) for i in line.split(' ')]
        
        lines = fp.readlines()
        self.__vectors = []
        self.__targets = []
        for line in lines:
            line.replace('\n', '')
            if re.match('in: ', line):
                line = re.sub('in: ', '', line)
                self.__vectors.append([float(i) for i in line.split(' ')])
            elif re.match('out: ', line):
                line = re.sub('out: ', '', line)
                self.__targets.append([float(i) for i in line.split(' ')])
            else:
                message = 'Error: invalid line'
                sys.exit(message)
        
        self.__pos = 0
        
    def isEof(self):
        
        if len(self.__vectors) == self.__pos:
            return True
        else:
            return False
            
    def getTopology(self):
        
        return self.topology
        
    def getNextValue(self):
        
        inputs = self.__vectors[self.__pos]
        target = self.__targets[self.__pos]
        self.__pos += 1
        
        return inputs, target
