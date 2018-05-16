#!/usr/bin/env python3.6
import numpy as np
import network
from tabulate import tabulate
import string
import cv2
import random
from pathlib import Path
from collections import namedtuple

Train = namedtuple('Train', ('char', 'input'))

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


def lesson(char, expect):
    """
    Вызывает обучение, сравнивет полученый результат с ожидаемым.
    Если в отданым функцией обучения массива максимальное число стоит на том же месте что и максимальное число в ожидаемом, то значение угадано правильно, иначе нет.

    Parametrs
    ---------
    char : numpy.ndarray
        Двумерный массив с символом для распознавания
    expect : list
        Массив с отмеченой правильной буквой

    Returns
    -------
    bool
        Если угаданое значение правильно -- `True`, иначе -- `False`
    """
    result = err(char, expect)
    return result.index(max(result)) == expect.index(max(expect))


def epoch(pics):
    """
    Вызывает lesson для каждой картинки в обучающей выборке, собирает ожидаемое значение.
    В массиве ожидаемого значения на месте правильной буквы стоит 1, иначе 0.

    Parametrs
    ---------
    pics : list
        Список из namedtuple `Train`

    Yields
    ------
    bool
        Результат работы функции lesson
    """
    for pic in pics:
        yield lesson(pic.input,
                     list(map(lambda x: x == pic.char, string.ascii_letters)))


def load(pics):
    """
    Загрузка всей обучающей выборки

    Parametrs
    ---------
    pics : list
        Список файлов источников, элементы -- `Path`
    Yields
    ------
    Train
        namedtuple,
        char -- символ,
        input -- значнения, изображение, приведенное к необходимому для нейроной сети виду
    """
    for pic in pics:
        yield Train(char=pic.stem[0], input=(255 - cv2.imread(str(pic), 0)) / 255)


def learn():
    """
    Вызывает функции лоя обучения, перемешивает обучающую выборку после каждой эпохи
    """
    pics = list(load(Path().glob(r'pic/[a-zA-Z]*.png')))
    for i in range(10):
        random.shuffle(pics)
        right = sum(epoch(pics))
        print(f'{i}: {right}')

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
