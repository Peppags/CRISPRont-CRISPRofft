# -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.initializers import RandomUniform
from keras.layers import multiply
from keras.layers.core import Reshape, Permute
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Lambda, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.models import *
import numpy as np
import pandas as pd


def attention(x, g, TIME_STEPS):
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
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


def loadData(data_file):
    data_list, data = [], []
    with open(data_file) as f:
        for line in f:
            l = [i for i in line.strip().split(',')]
            data_item = [int(i) for i in l[2:]]
            data.append(l)
            data_list.append(data_item)
    return data


VOCAB_SIZE = 16
EMBED_SIZE = 90
MAXLEN = 23


def crispr_offt():
    input = Input(shape=(23,))
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

    conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)

    conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)

    conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
    batchnor3 = BatchNormalization()(conv3)

    conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
    x = Lambda(lambda x: attention(x[0], x[1], 11))([conv11, batchnor3])

    flat = Flatten()(x)
    dense1 = Dense(40, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(20, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model


if __name__ == '__main__':
    model = crispr_offt()

    print("Loading weights for the models")
    model.load_weights("weights/CRISPR-OFFT.h5")

    sgrna = input("\nInput the sgRNA sequence followed by the PAM sequence(23 base pair sequence):\n")
    dna = input("\nInput the DNA sequence:\n")
    proceed_flag = 1

    if len(sgrna) < 23 or len(dna) < 23:
        print("Sequence is too short.")
        proceed_flag = 0

    if sgrna.count('A') + sgrna.count('T') + sgrna.count('C') + sgrna.count('G') != len(sgrna):
        print("The sgRNA sequence should contains four characters A, T, C and G.")
        proceed_flag = 0

    if dna.count('A') + dna.count('T') + dna.count('C') + dna.count('G') != len(dna):
        print("DNA sequence should contains four characters A, T, C and G.")
        proceed_flag = 0

    if len(sgrna) > 23 or len(dna) > 23:
        print("Sequence is too long.")
        proceed_flag = 0

    if proceed_flag == 1:
        sgrna = pd.DataFrame(np.array(sgrna).reshape(-1))
        dna = pd.DataFrame(np.array(dna).reshape(-1))
        offt_data = pd.concat([sgrna, dna], axis=1)

        offt_data_path = "data/test_offt.csv"
        offt_data.to_csv(offt_data_path, index=False, sep=',', header=['sgRNA', 'DNA'])

        offt_data = pd.read_csv(offt_data_path)

        match_dic = {
                      "AA": 1, "AC": 2, "AG": 3, "AT": 4,
                      "CA": 5, "CC": 6, "CG": 7, "CT": 8,
                      "GA": 9, "GC": 10, "GG": 11, "GT": 12,
                      "TA": 13, "TC": 14, "TG": 15, "TT": 16
                    }

        sgrna_list = offt_data['sgRNA'].values
        dna_list = offt_data['DNA'].values

        encoded_file = "data/encoded_test_offt.txt"
        encoded_data = open(encoded_file, 'w')
        length = len(offt_data)
        for i in range(length):
            vector = []
            vector.append(sgrna_list[i])
            vector.append(dna_list[i])
            for j in range(len(sgrna_list[i])):
                temp = sgrna_list[i][j] + dna_list[i][j]
                vector.append(match_dic[temp] - 1)
            encoded_data.writelines(",".join('%s' % item for item in vector) + '\n')
        encoded_data.close()

        xtest = loadData(encoded_file)
        xtest = np.array(xtest)

        print("Predicting on test data")
        y_pred = model.predict(xtest[:, 2:])
        max_index = np.argmax(y_pred)
        if max_index == 0:
            print('The input sequence belongs to non-off-target with possibility %.4f' % y_pred[:, 0])
        else:
            print('The input sequence belongs to off-target with possibility %.4f' % y_pred[:, 1])

