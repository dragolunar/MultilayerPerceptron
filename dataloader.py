# -*- coding: utf-8 -*-

import sys
import re

# Silly class to read training/test data from a file
class DataLoader:
    
    def __init__(self, fileName, scaler=None):
        
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
            temp = []
            if re.match('in: ', line):
                line = re.sub('in: ', '', line)
                for i in line.split(' '):
                    if re.match('[0-9]+', i):
                        if scaler is not None:
                            temp.append(scaler(float(i)))
                        else:
                            temp.append(float(i))
                self.__vectors.append(temp)
            elif re.match('out: ', line):
                line = re.sub('out: ', '', line)
                for i in line.split(' '):
                    if re.match('[0-9]+', i):
                        temp.append(float(i))
                self.__targets.append(temp)
            else:
                message = 'Error: invalid line'
                sys.exit(message)
        
        self.__pos = 0
        fp.close()
        
    def isEof(self):
        
        if len(self.__vectors) == self.__pos:
            return True
        else:
            return False
            
    def getTopology(self):
        
        return self.__topology
        
    def getNextValues(self):
        
        inputs = self.__vectors[self.__pos]
        targets = self.__targets[self.__pos]
        self.__pos += 1
        
        return inputs, targets
        
    def head(self):
        
        self.__pos = 0
