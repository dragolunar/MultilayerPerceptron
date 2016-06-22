# -*- coding: utf-8 -*-

import sys
import struct
import dataloader
import network


def showVector(label, vector):
    message = label
    for val in vector:
        message += str(val) + " "
    print(message)

def scaling(x):
    
    # mini-max scaling
    return (x - 0)/(255 - 0)


if __name__ == '__main__':
    
    MAX_EPOCH = 100
    LEARNING_RATE = 0.15
    LEAST_ERROR = 0.001
    
    #trainData = dataloader.DataLoader('trainingData.txt')
    trainData = dataloader.DataLoader('samples_1000.txt', scaling)
    
    topology = trainData.getTopology()
    myNet = network.Network(topology, LEARNING_RATE)
    
    print('')
    print('Training:')
    print('')
    
    epochs = 0
    error = 1.0
    while (epochs < MAX_EPOCH) and (error > LEAST_ERROR):
        
        epochs += 1
        trainData.head()
        print('Epoch: %d' % epochs)
        
        trainPass = 0
        while not trainData.isEof():
            
            trainPass += 1
            
            # Get new input data and feed it forward
            inputVector, targetVector = trainData.getNextValues()
            if (inputVector is None) or (targetVector is None):
                break
            
            myNet.feedForward(inputVector)
            
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
            print('Net recent average error: %f' % myNet.getRecentAverageError())
            '''
            
        error = myNet.getRecentError()
        print('Net error: %f' % error)
        
    print('')
    print('Training done!')
    
    #fp = open('weights.bin', 'wb')
    fp = open('weights.txt', 'w')
    fp.write('topology: ')
    for i in trainData.getTopology():
        out = str(i) + ' '
        fp.write(out)
    fp.write('\n')
    for n in range(len(myNet.layers)):
        for j in range(len(myNet.layers[n])):
            for k in range(len(myNet.layers[n][j].getWeights())):
                #fp.write(struct.pack('f', myNet.layers[n][j].weights[k].weight))
                out = '(' + str(n) + ',' + str(j) + ',' + str(k) + '): '
                out += str(myNet.layers[n][j].weights[k].weight)
                out += '\n'
                fp.write(out)
    fp.close()
