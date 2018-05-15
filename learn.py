#!/usr/bin/env python3.6
import numpy as np
import network
from tabulate import tabulate
import string
import cv2
import random

speed = .1


def p(s, table):
    print(s)
    print(tabulate(table, tablefmt='fancy_grid'))


def err(test, expect):
    first = list(network.first(test, network.activation))
    firstIn = list(network.first(test, lambda x: x))

    out = list(network.second(first, network.activation))
    outIn = list(network.second(first, lambda x: x))

    error = np.subtract(expect, out) * \
        [network.activationDer(x) for x in outIn]
    delta2 = speed * np.array([np.multiply(first, x) for x in error])

    error_in = np.array([np.sum(error * x) for x in network.layer2.T])
    error2 = error_in * [network.activationDer(x) for x in firstIn]
    delta = speed * np.array([np.multiply(test, x) for x in error2])

    network.layer2 += delta2
    network.layer1 += delta

    delta2Bias = speed * error
    deltaBias = speed * error2
    network.bias2 += delta2Bias
    network.bias1 += deltaBias

    return out


def testLetter(letter):
    letterNP = np.abs((255 - cv2.imread('pic/' + letter + '.png', 0)) / 255)
    return err(letterNP, list(map(lambda x: x == letter,
                                  string.ascii_letters)))


def oneTest(letter, letters):
    res = testLetter(letter)
    index = letters.index(letter)
    # print(letter, letters[res.index(max(res))], res[index], max(res), res.index(max(res)))
    return res[index] >= max(res)


def learn():
    # a = testLetter()[0] > .7
    # print(a)
    # print(list(map(lambda x: x == 'A', string.ascii_uppercase)))
    seq = list(string.ascii_letters)
    for i in range(400):
        random.shuffle(seq)
        a = sum([oneTest(x, string.ascii_letters) for x in seq])
        print(f'{a},')


# q = cv2.imread('pic/A.png', 0)
# q = np.abs((255 - q) / 255)
# print(q)

# a = err(q, list(map(lambda x: x == 'A', string.ascii_uppercase)))
# print(a)

# a = sum([err(q, list(map(lambda x: x == 'A', string.ascii_uppercase)))
#          [0] > 0.7 for x in range(100)])
# print(a)

learn()

np.save('./f/layer1.npy', network.layer1)
np.save('./f/layer2.npy', network.layer2)
np.save('./f/bias1.npy', network.bias1)
np.save('./f/bias2.npy', network.bias2)
