# -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
KTF.set_session(session)

from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Input, Embedding, Lambda, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import *
import numpy as np
import pandas as pd


def make_data(X):
    vectorizer = text.Tokenizer(lower=False, split=" ", num_words=None, char_level=True)
    vectorizer.fit_on_texts(X)
    alphabet = "ATCG"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    word_index = {k: (v+1) for k, v in char_dict.items()}
    word_index["PAD"] = 0
    word_index["START"] = 1
    vectorizer.word_index = word_index.copy()
    X = vectorizer.texts_to_sequences(X)
    X = [[word_index["START"]] + [w for w in x] for x in X]
    X = sequence.pad_sequences(X)
    return X


def AttentionLayer(inputs):
    time_steps = inputs.shape[1].value
    W = K.random_uniform_variable(shape=(time_steps, time_steps), low=0, high=1)
    b = K.random_uniform_variable(shape=(time_steps, ), low=0, high=1)
    x = K.permute_dimensions(inputs, (0, 2, 1))
    x1 = Lambda(lambda x: K.dot(x, W) + b)(x)
    x2 = Lambda(lambda x: K.tanh(x))(x1)
    a = K.softmax(x2)
    outputs = Lambda(lambda x: x[0] * x[1])([a, x])
    outputs = K.permute_dimensions(outputs, (0, 2, 1))
    return outputs


def main():
    dropout_rate = 0.3
    input = Input(shape=(24,))
    embedded = Embedding(7, 44, input_length=24)(input)

    conv1 = Conv1D(256, 5, activation='relu', name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)
    atten1 = Lambda(lambda x: AttentionLayer(x))(batchnor1)
    drop1 = Dropout(dropout_rate)(atten1)

    conv2 = Conv1D(128, 5, activation='relu', name="conv2")(drop1)
    batchnor2 = BatchNormalization()(conv2)
    atten2 = Lambda(lambda x: AttentionLayer(x))(batchnor2)
    drop2 = Dropout(dropout_rate)(atten2)

    flat = Flatten()(drop2)

    dense1 = Dense(128, activation='relu', name="dense1")(flat)
    batchnor3 = BatchNormalization()(dense1)
    drop3 = Dropout(dropout_rate)(batchnor3)

    dense2 = Dense(64, activation='relu', name="dense2")(drop3)
    batchnor4 = BatchNormalization()(dense2)
    drop4 = Dropout(dropout_rate)(batchnor4)

    dense3 = Dense(32, activation='relu', name="dense3")(drop4)
    batchnor5 = BatchNormalization()(dense3)
    drop5 = Dropout(dropout_rate)(batchnor5)

    output = Dense(1, activation='linear', name="output")(drop5)

    model = Model(inputs=[input], outputs=[output])

    print("Loading weights for the models")
    model.load_weights('weights/A_CNNCrispr_weights.h5')

    print("Loading test data")
    test_file = "data/test_data.csv"
    data = pd.read_csv(test_file, usecols=[0, 1])
    data = np.array(data)
    x_data, y_test = data[:, 0], data[:, 1]

    x_test = make_data(x_data)
    y_test = y_test.reshape(len(y_test), -1)

    print("Predicting on test data")
    result_file = "result/result.csv"
    y_test = pd.DataFrame(y_test)
    y_pred = model.predict([x_test], batch_size=256, verbose=2)
    y_pred = pd.DataFrame(y_pred)

    result = pd.concat([y_test, y_pred], axis=1)
    result.to_csv(result_file, index=False, sep=',', header=['y_test', 'y_pred'])


if __name__ == '__main__':
    main()