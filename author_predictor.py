from keras.models import Sequential
from keras.layers import SimpleRNN
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

    with open(fname) as f:
        raw = f.read()
        raw = raw.lower()

        chars = sorted(list(set(raw)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))

        n_chars = len(raw)
        n_vocab = len(chars)
        print "Total Characters: ", n_chars
        print "Total Vocab: ", n_vocab

        char_count = 0

        for i in range(0, n_chars - distance, 1):
            sequences_in = raw[i:i + distance]
            sequences_out = raw[i + distance]
            input.append([char_to_int[char] for char in sequences_in])
            expected_output.append(char_to_int[sequences_out])

        print len(input)

        return (input, expected_output)

(input, categories) = load_book(authors[0]["book"])

X = np.reshape(input, (len(input),len(input[0]),1))
X = X / float(len(categories))
Y = np_utils.to_categorical(categories)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_list)
