#!/usr/bin/env python3.6
import cv2
from PIL import Image
import numpy as np
import start
import pic
import sys

II = 16
SPACE = 5132
NEWLINE = 5130


def center(image):
    """ center(image)
    Переводит изображение из прямоугольника в квадрат, по большей стороне,
    с сохранением пропорций.

    Parametrs
    ---------
    image : numpy.ndarray
        Изображение

    Returns
    -------
    numpy.ndarray
        Квадратное изобажение
    """
    h, w = image.shape
    x, y = (h, h) if h > w else (w, w)
    square = np.ones((x, y), np.uint8) * 255
    square[int((y - h) / 2):int(y - (y - h) / 2),
           int((x - w) / 2):int(x - (x - w) / 2)] = image
    return square


def scale(image, imagesize):
    """ scale(image)
    Скалирует изображение до imagesize на imagesize, с сохранением пропорций.

    Parametrs
    ---------
    image : numpy.ndarray
        Изображение
    imagesize : int
        Размар будущего изображения

    Returns
    -------
    numpy.ndarray
        Изображение размерами imagesize на imagesize
    """
    return cv2.resize(center(image), (imagesize, imagesize))
    # return np.array(pic.scale(Image.fromarray(image), (II, II)))


def lines(thresh):
    """ lines(thresh)
    Осуществляет поиск строк на бинаризированном изображении.
    Ищет горизонтальную строку не имеющую черных пикселей.

    Parametrs
    ---------
    thresh : numpy.ndarray
        2d array, бинаризированное изображение

    Yields
    ------
    numpy.ndarray
        Следующая найденая строка
    """
    hist = np.sum(thresh, axis=1) // 255
    last = 0
    line = 0
    for j, i in enumerate(hist):
        if i == 0:
            if last < j - 1:
                yield thresh[last:j]
            last = j
        if i != 0:
            if j - line > 40:
                yield NEWLINE
            line = j


def chars(thresh):
    """ chars(thresh)
    Осуществляет поиск букв на бинаризированном изображении.
    Ищет вертикальную строку не имеющую черных пикселей.

    Parametrs
    ---------
    thresh : numpy.ndarray
        2d array, бинаризированное изображение

    Yields
    ------
    numpy.ndarray
        Следующая найденая буква
    """
    hist = np.sum(thresh, axis=0) // 255
    end = len(hist)
    last = 0
    word = 0
    for j, i in enumerate(hist):
        if i == 0:
            if last < j - 1:
                yield thresh[:, last:j]
            last = j
        if i != 0:
            if j - word > 40:
                yield SPACE
            word = j
        if j == end - 1:
            yield SPACE


def findChar(image):
    """ findChar(image)
    Осуществляет поиск букв на кртинке.
    Разделяет ее сначала на строки, а потом на буквы.

    Parametrs
    ---------
    image : numpy.ndarray
        OpenCV изображение

    Yields
    ------
    numpy.ndarray
        OpenCV бинаризированное изображение буквы
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    for line in lines(thresh):
        if line is NEWLINE:
            yield NEWLINE
            continue
        for char in chars(line):
            if char is SPACE:
                yield SPACE
                continue
            thresh = cv2.morphologyEx(char, cv2.MORPH_CLOSE, kernel)
            _, contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ans = []
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                if h < 60:
                    continue
                # yield scale(255 - char[y:y + h, x:x + w], II)
                ans.append((scale(255 - char[y:y + h, x:x + w], II), x))
            ans = map(lambda x: x[0], sorted(ans, key=lambda x: x[1]))
            yield from ans


if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])
    ans = []
    for char in findChar(image):
        if char is SPACE:
            ans.append(' ')
            continue
        if char is NEWLINE:
            ans.append('\n')
            continue
        qwe = (255 - char) / 255
        ans.append(start.start(qwe))
    print(''.join(ans))
    print('======')
