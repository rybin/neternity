#!/usr/bin/env python3.6
import string
import cv2
import numpy as np
from pathlib import Path
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten

nums = 52

inp = []
char = []
for pic in Path().glob(r'pic/[a-zA-Z]*.png'):
    # inp.append(np.array((255 - cv2.imread(str(pic), 0)) / 255).reshape(1, 256)[0])
    inp.append((255 - cv2.imread(str(pic), 0)) / 255)
    char.append(string.ascii_letters.index(pic.stem[0]))
    # char.append(list(map(lambda x: x == pic.stem[0], string.ascii_letters)))

inp = np.array(inp)

char = np.array(char)

char = keras.utils.to_categorical(char, nums)

model = Sequential()

model.add(Dense(units=128, activation='relu', input_shape=(16, 16)))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=nums, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(inp, char, epochs=50, batch_size=32)

model.save('mymodel')
