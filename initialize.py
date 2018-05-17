#!/usr/bin/env python3.6
import numpy as np

II = 16
SL = 80
OL = 52

# np.save('./f/bias1.npy', np.zeros(SL))
# np.save('./f/bias2.npy', np.zeros(OL))

np.save('./f/bias1.npy', np.random.rand(SL))
np.save('./f/bias2.npy', np.random.rand(OL))

# np.save('./f/layer1.npy', np.random.rand(SL, II, II) / 2 + 1 / 4)
# np.save('./f/layer2.npy', np.random.rand(OL, SL) / 2 + 1 / 4)

np.save('./f/layer1.npy', np.random.rand(SL, II, II) - 0.5)
np.save('./f/layer2.npy', np.random.rand(OL, SL) - 0.5)

# np.save('./f/layer1.npy', np.zeros((SL, II, II)))
# np.save('./f/layer2.npy', np.zeros((OL, SL)))
