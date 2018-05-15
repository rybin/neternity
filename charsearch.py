#!/usr/bin/env python3.6
import cv2
from PIL import Image
import numpy as np
import start
import pic


def findChar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 10)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow('image', thresh)
    # cv2.waitKey(0)
    # return False
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        # cv2.rectangle(i, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # cv2.imshow('image', i[y:y + h, x:x + w])
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
    for i, x, y in findChar(image):
        # print(start.start(np.abs((255 - i) / 255)))
        ans.append((start.start(np.abs((255 - i) / 255)), x, y))
    # print(ans)
    ans = np.array(sorted(ans, key=lambda x: x[1]))
    ans = ' '.join(ans[:, 0])
    print(ans)

    # image = findWord(image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
