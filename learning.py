# -*- coding: utf-8 -*-

import sys
import dataloader
import network


def showVector(label, vector):
    message = label + " "
    for val in vector:
        message += str(val) + " "
    print(message)


if __name__ == '__main__':
    
    trainData = dataloader.DataLoader('trainingData.txt')

    topology = trainData.getTopology()
    myNet = network.Network(topology)

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
        print('Delta weights: %f' % myNet.layers[-1][0].weights[0].deltaWeight)

        # Report how well the training is working, averaged over recent samples
        message = 'Net recent average error:'
        message += str(myNet.getRecentAverageError())
        print(message)

    print('')
    print('Done')
