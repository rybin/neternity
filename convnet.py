#!/usr/bin/env python3.6
import numpy as np

kernel = np.load('./f/kernel.npy')


def kfunc(part):
    return np.sum(part * kernel)


def convlayer(inpm):
    result = np.zeros((inpm.shape[0] - kernel.shape[0] + 1,
                       inpm.shape[1] - kernel.shape[1] + 1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j] = kfunc(inpm[i:i + len(kernel), j:j + len(kernel)])
    return result


def pooling(inpm):
    result = np.zeros((inpm.shape[0] // 2, inpm.shape[1] // 2))
    for i in range(0, inpm.shape[0], 2):
        for j in range(0, inpm.shape[1], 2):
            result[i // 2, j // 2] = np.sum(inpm[i:i + 2, j:j + 2])
    return result


def call(inp):
    return pooling(convlayer(inp))
