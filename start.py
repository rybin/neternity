#!/usr/bin/env python3.6
import cv2
import sys
import network
import string
import numpy as np

p = string.ascii_letters


def start(inp):
    a = list(network.call(inp))
    q = a.index(max(a))
    return p[q]


if __name__ == '__main__':
    inp = np.abs((255 - cv2.imread(sys.argv[1], 0)) / 255)
    print(start(inp))
