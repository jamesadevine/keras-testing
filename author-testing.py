from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense
from keras.utils import np_utils
import numpy as np
from numpy import float32

model_json = ''

with open('model.json') as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights('weights-improvement-18-3.0437.hdf5')

model.evaluate(input())
