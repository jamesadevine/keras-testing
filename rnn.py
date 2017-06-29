from keras.models import Sequential
from keras.layers import SimpleRNN
import numpy as np
from numpy import float32
import csv

def read_csv(fname):

    input = []
    labels = []
    with open(fname) as f:
        for line in f.readlines():
            if len(line) < 10:
                continue

            line = line.strip("\r\n")
            l = line.split(",")
            input += [float32(x) for x in l[0:4]]
            labels += [float32(x) for x in l[4:]]

        return {"labels":np.array(labels,dtype=float32),"input":np.array(input,dtype=float32)}


test_data = read_csv("irisTestData.txt")
test_data["input"] = test_data["input"].reshape(30,4,1)
training_data = read_csv("irisTrainData.txt")

model = Sequential()

print test_data["input"].reshape(30,4)

rnn = SimpleRNN(3, activation='tanh', input_shape=(4,1))

model.add(rnn)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print len(test_data["input"])
print test_data["input"]

#test_data["labels"]
model.fit(test_data["input"], test_data["labels"].reshape(30,3), batch_size=30, verbose=1)
