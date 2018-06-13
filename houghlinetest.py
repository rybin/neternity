#!/usr/bin/env python3.6
import cv2
import numpy as np

img = cv2.imread('testr.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
# cv2.imshow('img', edges)
# cv2.waitKey()
# exit()

lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
ttt = []
for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # print(rho, theta)
        # print(a, b)
        # ttt.append(theta)
        ttt = theta
    break

# cv2.imwrite('houghlines3.jpg', img)

print(ttt)
ttt = 90 - np.rad2deg(ttt)
# exit()

M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), -ttt, 1)
img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))


def lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    a = np.sum(thresh, axis=1)
    # print(a)
    maxx = thresh.shape[1] / 100
    last = 0
    for j, i in enumerate(a):
        if i < maxx:
            if last < j - 1:
                yield image[last:j]
            last = j
            image[j] = 0, 255, 0

list(lines(img))

cv2.imshow('img', img)
cv2.waitKey()
exit()