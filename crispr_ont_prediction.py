# -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.preprocessing import text, sequence
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv1D, AveragePooling1D
from keras.layers import multiply
from keras.layers.core import Dense, Reshape, Lambda, Permute, Flatten
from keras.initializers import RandomUniform
from keras import regularizers
import keras.backend as K
import keras
import numpy as np
import pandas as pd


def make_data(X):
    vectorizer = text.Tokenizer(lower=False, split=" ", num_words=None, char_level=True)
    vectorizer.fit_on_texts(X)
    alphabet = "ATCG"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    word_index = {k: (v + 1) for k, v in char_dict.items()}
    word_index["PAD"] = 0
    word_index["START"] = 1
    vectorizer.word_index = word_index.copy()
    X = vectorizer.texts_to_sequences(X)
    X = [[word_index["START"]] + [w for w in x] for x in X]
    X = sequence.pad_sequences(X)
    return X


def attention(x, g, TIME_STEPS):
    input_dim = int(x.shape[2])
    x1 = K.permute_dimensions(x, (0, 2, 1))
    g1 = K.permute_dimensions(g, (0, 2, 1))

    x2 = Reshape((input_dim, TIME_STEPS))(x1)
    g2 = Reshape((input_dim, TIME_STEPS))(g1)
    x3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))(x2)
    g3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))(g2)
    x4 = keras.layers.add([x3, g3])
    a = Dense(TIME_STEPS, activation="softmax", use_bias=False)(x4)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([x, a_probs])
    return output_attention_mul


def crispr_ont():
    dropout_rate = 0.4
    input = Input(shape=(24,))
    embedded = Embedding(7, 44, input_length=24)(input)

    conv1 = Conv1D(256, 5, activation="relu", name="conv1")(embedded)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(dropout_rate)(pool1)

    conv2 = Conv1D(256, 5, activation="relu", name="conv2")(pool1)

    conv3 = Conv1D(256, 5, activation="relu", name="conv3")(drop1)

    x = Lambda(lambda x: attention(x[0], x[1], 6))([conv3, conv2])

    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    weight_1 = Lambda(lambda x: x * 0.2)
    weight_2 = Lambda(lambda x: x * 0.8)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(x)
    flat = my_concat([weight_1(flat1), weight_2(flat2)])

    dense1 = Dense(128,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu",
                   name="dense1")(flat)
    drop3 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(64,
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu",
                   name="dense2")(drop3)
    drop4 = Dropout(dropout_rate)(dense2)

    dense3 = Dense(32, activation="relu", name="dense3")(drop4)
    drop5 = Dropout(dropout_rate)(dense3)

    output = Dense(1, activation="linear", name="output")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model


if __name__ == '__main__':
    model = crispr_ont()

    print("Loading weights for the models")
    model.load_weights("crispr_ont.h5")

    spacer_pam = input("\nInput the sgRNA sequence followed by the PAM sequence(23 base pair sequence):\n")
    proceed_flag = 1

    if len(spacer_pam) < 23:
        print("Sequence is too short.")
        proceed_flag = 0

    if spacer_pam.count('A') + spacer_pam.count('T') + spacer_pam.count('C') + spacer_pam.count('G') != len(spacer_pam):
        print("Sequence should contains four characters A, T, C and G.")
        proceed_flag = 0

    if len(spacer_pam) > 23:
        print("Sequence is too long.")
        proceed_flag = 0

    if proceed_flag == 1:
        spacer_pam = pd.DataFrame(np.array(spacer_pam).reshape(-1))
        data_path = "data/test_ont.csv"
        spacer_pam.to_csv(data_path, index=False, sep=',', header=['sgRNA'])

        data = pd.read_csv(data_path)
        x_test = make_data(data["sgRNA"])

        print("Here is the cleavage efficiency that CRISPR-ONT predicts for this guide:")
        y_pred = model.predict([x_test], batch_size=128, verbose=2)
        print("%.2f" % y_pred[0][0])

