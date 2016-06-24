# -*- coding: utf-8 -*-

import sys
import dataloader
import network


def showVector(label, vector):
    message = label
    for val in vector:
        message += str(val) + " "
    print(message)

def scaling(x):
    
    # mini-max scaling
    result = [(i - 0)/(255 - 0) for i in x]
    return result


if __name__ == '__main__':
    
    MAX_EPOCH = 1000
    LEARNING_RATE = 0.15
    LEAST_AVG_ERROR = 0.001
    
    #trainData = dataloader.InputData('trainingData.txt')
    trainData = dataloader.InputData('samples_1000.txt')
    
    topology = trainData.getTopology()
    sampleSize = trainData.getSampleSize()
    myNet = network.Network(topology, LEARNING_RATE)
    
    print('')
    print('Training:')
    print('')
    
    epochs = 0
    while epochs < MAX_EPOCH:
        
        epochs += 1
        trainData.head()
        print('Epoch: %d' % epochs)
        
        recentAvgError = 0.0
        totalError = 0.0
        trainPass = 0
        while not trainData.isEof():
            
            trainPass += 1
            
            # Get new input data and feed it forward
            inputVector, targetVector = trainData.getNextValues()
            if (inputVector is None) or (targetVector is None):
                break
            
            myNet.feedForward(scaling(inputVector))
            totalError += myNet.getRecentError()
            recentAvgError = (recentAvgError*sampleSize + myNet.getRecentError())/(sampleSize + 1.0)
            
            # Collect the net's actual results
            outputVector = myNet.getResults()
            
            # Train the net what the outputs should
            if len(targetVector) != topology[-1]:
                message = 'Error: dimension of output vector is not match!'
                sys.exit(message)
            myNet.backPropagation(targetVector)
            
            # Report how well the training is working, averaged over recent samples
            '''
            print('')
            print('Pass: %d' % trainPass)
            showVector('Inputs: ', inputVector)
            showVector('Outputs: ', outputVector)
            showVector('Targets: ', targetVector)
            print('Net recent average error: %f' % recentAvgError)
            '''
            
        avgError = totalError/sampleSize
        print('Net error: %f' % avgError)
        if avgError < LEAST_AVG_ERROR:
            break
        
    print('')
    print('Training done!')
    
    fp = open('weights.txt', 'w')
    fp.write('topology: ')
    for i in trainData.getTopology():
        out = str(i) + ' '
        fp.write(out)
    fp.write('\n')
    
    layers = myNet.getLayers()
    for n in range(len(layers)):
        neurons = layers[n]
        for k in range(len(neurons)):
            weights = neurons[k].getWeights()
            for j in range(len(weights)):
                out = str(n) + ' ' + str(k) + ' ' + str(j) + ' '
                out += str(weights[j].weight) + '\n'
                fp.write(out)
    fp.close()
