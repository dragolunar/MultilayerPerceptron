# -*- coding: utf-8 -*-

import sys
import struct
import dataloader
import network


def showVector(label, vector):
    message = label + " "
    for val in vector:
        message += str(val) + " "
    print(message)

def scaling(x):
    
    # mini-max scaling
    return (x - 0)/(255 - 0)


if __name__ == '__main__':
    
    MAX_EPOCH = 1
    LEARNING_RATE = 0.15
    
    trainData = dataloader.DataLoader('trainingData.txt')
    #trainData = dataloader.DataLoader('samples_1000.txt', scaling)

    topology = trainData.getTopology()
    myNet = network.Network(topology, LEARNING_RATE)

    epochs = 0
    while not trainData.isEof():
        epochs += 1
        print('')
        print('Epoch: %d' % epochs)

        # Get new input data and feed it forward
        inputVector, targetVector = trainData.getNextValues()
        if (inputVector is None) or (targetVector is None):
            break
        
        showVector('Inputs:', inputVector)
        myNet.feedForward(inputVector)

        # Collect the net's actual results
        outputVector = myNet.getResults()
        showVector('Outputs:', outputVector)

        # Train the net what the outputs should
        showVector("Targets:", targetVector)
        if len(targetVector) != topology[-1]:
            message = 'Error: dimension of output vector is not match!'
            sys.exit(message)

        myNet.backPropagation(targetVector)

        # Report how well the training is working, averaged over recent samples
        message = 'Net recent average error:'
        message += str(myNet.getRecentAverageError())
        print(message)

    print('')
    print('Training done!')
    
    
    fp = open('weights.bin', 'wb')
    for layer in myNet.layers:
        for neuron in layer:
            for connection in neuron.weights:
                fp.write(struct.pack('f', connection.weight))
    fp.close()
