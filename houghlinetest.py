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

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        print(rho, theta)
        print(a, b)
        ttt = theta
    break

# cv2.imwrite('houghlines3.jpg', img)

M = cv2.getRotationMatrix2D((img.shape[0] // 2, img.shape[1] // 2), -ttt, 1)
img = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))

cv2.imshow('img', img)
cv2.waitKey()
exit()
