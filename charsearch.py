#!/usr/bin/env python3.6
import cv2
from PIL import Image
import numpy as np
import start
import pic

II = 32


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
    hist = np.sum(thresh, axis=1)
    last = 0
    for j, i in enumerate(hist):
        if i < 1000:
            if last < j - 1:
                yield thresh[last:j]
            last = j


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
    hist = np.sum(thresh, axis=0)
    last = 0
    for j, i in enumerate(hist):
        if i == 0:
            if last < j - 1:
                yield thresh[:, last:j]
            last = j


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
    for line in lines(thresh):
        for char in chars(line):
            _, contours, _ = cv2.findContours(
                char, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                [x, y, w, h] = cv2.boundingRect(contour)
                yield scale(255 - char[y:y + h, x:x + w], II)


def findChar2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 10)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)
    # return False
    # print(hierarchy)
    # cv2.drawContours(image, contours, -1, (0,255,0), 3)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        # cv2.rectangle(i, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # cv2.imshow('image', image[y:y + h, x:x + w])
        # cv2.imshow('image', pic.scale(i[y:y + h, x:x + w], (32, 32)))
        # cv2.imwrite('c/'+str(np.random.rand()), np.array(pic.scale(Image.fromarray(i[y:y + h, x:x + w]), (32, 32))))
        yield np.array(pic.scale(Image.fromarray(image[y:y + h, x:x + w]), (32, 32))), x, y
        # cv2.imshow('image', np.array(pic.scale(Image.fromarray(image[y:y + h, x:x + w]), (32, 32))))
        # cv2.waitKey(0)
        # return False
    # cv2.imshow('image', image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    image = cv2.imread('test/testsample6.png')
    ans = []
    for char in findChar(image):
        qwe = char / 255
        # print(qwe[1, :])
        # print(char[1, :])
        # print(start.start(np.abs(255 - char) / 255))
        # print(start.start(qwe))
        # cv2.imshow('image', qwe)
        # cv2.waitKey(0)
        # cv2.imwrite(str(start.start(qwe)) + str(np.random.rand()) + '_1.png', char)
        ans.append(start.start(qwe))
        # print(char[1,:])
        # break
    print(ans)

    ans=[]
    for i, x, y in findChar2(image):
        # print(start.start(np.abs((255 - i) / 255)))
        # cv2.imshow('image', i)
        # cv2.waitKey(0)
        # cv2.imwrite(str(start.start(np.abs((255 - i) / 255))) + str(np.random.rand()) + '_2.png', i)
        ans.append((start.start(np.abs((255 - i) / 255)), x, y))
        # if start.start(np.abs((255 - i) / 255)) == 'D':
        #     # print(i[1,:])
        #     cv2.imwrite(str(start.start(np.abs((255 - i) / 255))) + str(np.random.rand()) + '_2.png', i)
        #     break
        # break
    # print(ans)
    ans = np.array(sorted(ans, key=lambda x: x[1]))
    ans = ' '.join(ans[:, 0])
    print(ans)

    # image = findWord(image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
