# -*- coding: utf-8 -*-

import enum
import numpy as np
import matplotlib.pyplot as plt


class Label(enum.IntEnum):
    RED     = 1
    GREEN   = 2
    BLUE    = 3


def scaling(x):
    
    # mini-max scaling
    return (x - 0)/(255 - 0)


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


if __name__ == '__main__':
    
    testData = 'samples_100000.txt'
    weightData = 'weights.txt'
