from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense
from keras.utils import np_utils
import numpy as np
from numpy import float32
import csv

authors = [
    {
        "author":"Lewis Caroll",
        "book":"alice-in-wonderland.txt"
    },
    {
        "author":"Jane Austen",
        "book":"pride-and-prejudice.txt"
    },
    {
        "author":"Charles Dickens",
        "book":"a-tale-of-two-cities.txt"
    },
]

def load_book(fname, distance = 100):

    input = []
    expected_output = []

    with open(fname, encoding="utf8") as f:
        print("Loading book: ",fname)
        raw = f.read()
        raw = raw.lower()

        chars = sorted(list(set(raw)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))

        n_chars = len(raw)
        n_vocab = len(chars)
        print("Total Characters: ", n_chars)
        print("Total Vocab: ", n_vocab)

        char_count = 0

        for i in range(0, n_chars - distance, 1):
            sequences_in = raw[i:i + distance]
            sequences_out = raw[i + distance]
            input.append([char_to_int[char] for char in sequences_in])
            expected_output.append(char_to_int[sequences_out])

        print(len(input))

        return (np.array(input), np.array(expected_output))

(input1, categories1) = load_book(authors[0]["book"])
(input2, categories2) = load_book(authors[1]["book"])
(input3, categories3) = load_book(authors[2]["book"])

labels = [authors[0]["author"],authors[1]["author"],authors[2]["author"]]

# normalize between 0 and one.
input1 = input1 / float(len(categories1))
input2 = input2 / float(len(categories2))
input3 = input3 / float(len(categories3))

inputs = np.concatenate((input1, input2, input3))

X = np.array(inputs).reshape(len(inputs), len(inputs[0]),1)

label1 = [[1,0,0]] * input1.shape[0]
label2 = [[0,1,0]] * input2.shape[0]
label3 = [[0,0,1]] * input3.shape[0]
Y = np.concatenate((label1, label2, label3))

model = Sequential()
model.add(LSTM(3, input_shape=(X.shape[1], X.shape[2])))
model.compile(loss='mean_squared_error', optimizer='adam')

with open('model.json','w') as f:
    f.write(model.to_json())

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)

model.save_weights("final-model.hdf5")
