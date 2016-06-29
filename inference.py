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
    
    plt.title('scatterplot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0.0,0.7])
    plt.ylim([0.0,0.7])
    
    for i in range(len(samples)):
        xyY = rgb2xyy(scaling(samples[i]))
        if colors is not None:
            plt.scatter(xyY[0], xyY[1], c=colors[i])
        else:
            r = int(samples[i][0])
            g = int(samples[i][1])
            b = int(samples[i][2])
            plt.scatter(xyY[0], xyY[1], c='#%02x%02x%02x'%(r,g,b))
        #plt.pause(0.001)
        
    plt.show()

def calcConfMat(targets, outputs):
    
    #confMat = [[0]*len(labels)]*len(labels)    # error
    confMat = [[0 for j in range(len(targets[0]))] for k in range(len(targets[0]))]
    for i in range(len(targets)):
        idx_gt = targets[i].index(max(targets[i]))
        idx_est = outputs[i].index(max(outputs[i]))
        confMat[idx_gt][idx_est] += 1
    return confMat

def showConfMat(labels, targets, outputs):
    
    confMat = calcConfMat(targets, outputs)
    
    # show confusion matrix
    print('')
    out_str = '%-8s|' % ''
    for j in range(len(labels)):
        out_str += '%8s|' % labels[j]
    print(out_str)
    
    for j in range(len(confMat)):
        out_str = '%-8s|' % labels[j]
        for k in range(len(confMat[j])):
            out_str += '%8s|' % confMat[j][k]
        print(out_str)

def showReport(labels, targets, outputs):
    
    confMat = calcConfMat(targets, outputs)
    
    tp = [0]*len(labels)
    fn = [0]*len(labels)
    fp = [0]*len(labels)
    for j in range(len(confMat)):
        for k in range(len(confMat[j])):
            if j == k:
                tp[j] += confMat[j][k]
            else:
                fn[j] += confMat[j][k]
                fp[k] += confMat[j][k]
                
    # show report
    print('')
    out_str = '%-8s| precision|    recall|' % ''
    print(out_str)
    for j in range(len(labels)):
        precision = float(tp[j])/(float(tp[j]) + float(fp[j]))
        recall = float(tp[j])/(float(tp[j]) + float(fn[j]))
        out_str = '%-8s|%0.8f|%0.8f|' % (labels[j], precision, recall)
        print(out_str)
    avg_precision = float(sum(tp))/(float(sum(tp)) + float(sum(fp)))
    avg_recall = float(sum(tp))/(float(sum(tp)) + float(sum(fn)))
    out_str = 'average |%0.8f|%0.8f|' % (avg_precision, avg_recall)
    print(out_str)


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
    targets = []
    outputs = []
    while not testData.isEof():
        
        testPass += 1
        
        inputVector, targetVector = testData.getNextValues()
        myNet.feedForward(scaling(inputVector))
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
        #color = outputVector.index(max(targetVector))
        target = targetVector.index(max(targetVector))
        if color == 0:
            colors.append('r')
        elif color == 1:
            colors.append('g')
        elif color == 2:
            colors.append('b')
        if color != target:
            colors[testPass - 1] = 'w'
        inputs.append(inputVector)
        targets.append(targetVector)
        outputs.append(outputVector)
        
    labels = ['RED', 'GREEN', 'BLUE']
    showColorSpace(inputs, colors)
    showConfMat(labels, targets, outputs)
    showReport(labels, targets, outputs)
