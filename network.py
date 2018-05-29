#!/usr/bin/env python3.6
import numpy as np

layer1 = np.load('./f/layer1.npy')
layer2 = np.load('./f/layer2.npy')
bias1 = np.load('./f/bias1.npy')
bias2 = np.load('./f/bias2.npy')


def activation(put):
    """ Сигмоидальная функция активации

    Parametrs
    ---------
    put : int
        Полученное нейроном значение

    Returns
    -------
    int
        Возвращаемое нейроном значение
    """
    return 1 / (1 + np.exp(-put))


def activationDer(put):
    """ Производная сигмоидальной функции активации

    Parametrs
    ---------
    put : int

    Returns
    -------
    int
    """
    return activation(put) * (1 - activation(put))


def first(inp, act):
    """ Скрытый слой нейронной сети

    Parametrs
    ---------
    inp : numpy.ndarray
        Сигнал поданый через связи между входным и скрытым слое
    act : function
        Функция активации

    Yields
    ------
    int
        Выходное значение одного нейрона скрытого слоя
    """
    for weight, bias in zip(layer1, bias1):
        yield act(np.sum(np.multiply(weight, inp)) + bias)


def second(first, act):
    """ Выходной слой нейронной сети

    Parametrs
    ---------
    first : numpy.ndarray
        Сигнал поданый через связи между скрытым и выходным слое
    act : function
        Функция активации

    Yields
    ------
    int
        Выходное значение одного нейрона выходного слоя
    """
    for weight, bias in zip(layer2, bias2):
        yield act(np.sum(np.multiply(weight, first)) + bias)


def call(inp):
    """ Вызов нейронное сети

    Parametrs
    ---------
    inp : numpy.ndarray
        Входные сигналы

    Returns
    -------
    generator
        Генератор возвращающий выходные значения нейронной сети
    """
    return second(list(first(inp, activation)), activation)
