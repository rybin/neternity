#!/usr/bin/env python3.6
import numpy as np
import convnet

alpha = 1


def convlearn(test, error):
    dX = np.zeros(test.shape)
    dW = np.zeros(convnet.kernel.shape)
    out = convnet.convlayer(test)
    for i in range(error.shape[0]):
        for j in range(error.shape[1]):
            dX[i:i + dW[0], j:j + dW[1]] += convnet.kernel * error
            dW += test[i:i + dW[0], j:j + dW[1]] * error
