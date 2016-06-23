# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import dataloader
import network


def scaling(x):
    
    # mini-max scaling
    result = [(i - 0)/(255 - 0) for i in x]
    return result

def rgb2xyy(rgb):
    
    M = np.matrix([[ 0.412453, 0.357580, 0.180423],
                   [ 0.212671, 0.715160, 0.072169],
                   [ 0.019334, 0.119193, 0.950227]])
    
    XYZ = np.dot(M,rgb)
    X = XYZ[0,0]
    Y = XYZ[0,1]
    Z = XYZ[0,2]
    
    x = X/(X + Y + Z)
    y = Y/(X + Y + Z)
    #z = 1 - x - y
    xyY = np.array([x,y,Y])
    
    return xyY

def showVector(label, vector):
    
    message = label
    for val in vector:
        message += str(val) + " "
    print(message)

def showColorSpace(samples, colors=None):
    
    for i in range(len(samples)):
        xyY = rgb2xyy(scaling(samples[i]))
        if colors is not None:
            plt.scatter(xyY[0], xyY[1], c=colors[i])
        else:
            r, g, b = samples[i][0], samples[i][1], samples[i][2]
            plt.scatter(xyY[0], xyY[1], c='#%02x%02x%02x'%(r,g,b))
            
    plt.title('scatterplot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0.0,0.7])
    plt.ylim([0.0,0.7])
    plt.show()


if __name__ == '__main__':
    
    testData = 'samples_10000.txt'
    weightData = 'weights.txt'
    
    # Read test data
    #testData = dataloader.InputData(testData)
    testData = dataloader.InputData(testData)
    sampleSize = testData.getSampleSize()
    
    # Read weights data
    weightData = dataloader.WeightData(weightData)
    topology = weightData.getTopology()
    myNet = network.Network(topology)
    myNet.setWeights(weightData.getWeightValues())
    
    testPass = 0
    inputs = []
    colors = []
    while not testData.isEof():
        
        testPass += 1
        
        inputVector, targetVector = testData.getNextValues()
        myNet.feedForward(inputVector)
        outputVector = myNet.getResults()
        
        # Report how well the training is working, averaged over recent samples
        '''
        print('')
        print('Pass: %d' % testPass)
        showVector('Inputs: ', inputVector)
        showVector('Outputs: ', outputVector)
        showVector('Targets: ', targetVector)
        '''
        
        color = outputVector.index(max(outputVector))
        if color == 0:
            colors.append('r')
        elif color == 1:
            colors.append('g')
        else:
            colors.append('b')
        inputs.append(inputVector)
            
    showColorSpace(inputs, colors)
