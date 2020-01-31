# -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)
KTF.set_session(session)

from keras.preprocessing import sequence, text
from keras.models import Model
from keras.layers import Input, Embedding, Lambda, BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import *


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
    bio_input = Input(name='bio_input', shape=(2,))

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

    bio_dense1 = Dense(128, activation='relu', name="bio_dense1")(bio_input)
    bio_drop1 = Dropout(dropout_rate)(bio_dense1)

    bio_dense2 = Dense(64, activation='relu', name="bio_dense2")(bio_drop1)
    bio_drop2 = Dropout(dropout_rate)(bio_dense2)

    bio_dense3 = Dense(32, activation='relu', name="bio_dense3")(bio_drop2)
    bio_drop3 = Dropout(dropout_rate)(bio_dense3)

    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    merged = my_concat([drop5, bio_drop3])

    output = Dense(1, activation='linear', name="output")(merged)

    model = Model(inputs=[input, bio_input], outputs=[output])

    print("Loading weights for the models")
    weighs_path = "weights/A-CNNCrispr_Bio_weights.h5"
    model.load_weights(weighs_path)
    
    print("Loading test data")
    test_file = "data/test_data.csv"
    data = pd.read_csv(test_file, usecols=[0, 1, 2, 8])
    data = np.array(data)
    seq_test, y_test, bio_test = data[:, 0], data[:, 1], data[:, 2:]

    seq_test = make_data(seq_test)
    y_test = y_test.reshape(len(y_test), -1)

    bio_test = bio_test.reshape(len(bio_test), -1)
    scaler = MinMaxScaler()
    scaler.fit(bio_test)
    bio_test = scaler.transform(bio_test)

    y_test = y_test.reshape(len(y_test), -1)
    bio_test = bio_test.reshape(len(bio_test), -1)

    y_pred = model.predict([seq_test, bio_test], batch_size=8, verbose=2)

    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)

    print("Predicting on test data")
    result_file = "result/A-CNNCrispr_Bio_result.csv"
    y_pred_result = pd.concat([y_pred, y_test], axis=1)
    y_pred_result.to_csv(result_file, index=False, sep=',', header=['y_pred', 'y_test'])


if __name__ == '__main__':
    main()






