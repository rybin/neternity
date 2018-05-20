#!/usr/bin/env python3.6
import numpy as np

layer1 = np.load('./f/layer1.npy')
layer2 = np.load('./f/layer2.npy')
bias1 = np.load('./f/bias1.npy')
bias2 = np.load('./f/bias2.npy')


def activation(put):
    ''' Сигмоидальная функция активации '''
    return 1 / (1 + np.exp(-put))


def activationDer(put):
    ''' Производная сигмоидальной функции активации '''
    return activation(put) * (1 - activation(put))


# def activation(put):
#     ''' Функция активации '''
#     return 2 / (1 + np.exp(-put)) - 1


# def activationDer(put):
#     ''' Производная функции активации '''
#     return (1 / 2) * (1 + activation(put)) * (1 - activation(put))


def first(inp, act):
    for weight, bias in zip(layer1, bias1):
        yield act(np.sum(np.multiply(weight, inp)) + bias)


def second(first, act):
    for weight, bias in zip(layer2, bias2):
        yield act(np.sum(np.multiply(weight, first)) + bias)


def call(inp):
    return second(list(first(inp, activation)), activation)
