#!/usr/bin/env python3.6
import cv2
import numpy as np
import sys
import string
from keras.models import load_model

model = load_model('mymodel')

pic = sys.argv[1]
inp = np.array([(255 - cv2.imread(pic, 0)) / 255])
# inp = np.array((255 - cv2.imread(pic, 0)) / 255).reshape(1, 256)
# print(inp)

res = model.predict(inp, batch_size=32)[0]
res = res.tolist()
index = res.index(max(res))
print(index)
print(string.ascii_letters[index])
