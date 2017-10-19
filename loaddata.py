import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# 下記日間の傾向を見て、次の日の予測をする
LENGTH_OF_SEQUENCE = 30


def extractData(data, length_of_sequences):
    x, y = [], []
    for i in range(len(data) - (length_of_sequences + 1)):
        x.append(data.iloc[i:(i+length_of_sequences)].as_matrix())
        y.append(data.iloc[i+length_of_sequences+1].as_matrix())
    return (np.asarray(x, 'float'), np.asarray(y, 'float'))

def loadCsvData(filename):
    csvdata = pd.read_csv(filename)

    max, min = math.floor(csvdata[['end']].max()), math.floor(csvdata[['end']].min())
    x, y = extractData(data=csvdata[['end']], length_of_sequences=LENGTH_OF_SEQUENCE)
    x = normalize(max=max, min=min, data=x)
    y = normalize(max=max, min=min, data=y)
    return x.tolist(), y.tolist()

def normalize(max, min, data):
    ceil = max - min

    data -= min
    data /= ceil

    return data

def originalize(max, min, data):
    ceil = max - min

    data *= ceil
    data += min

    return data

def loadCsvDataXOnly(filename):
    csvdata = pd.read_csv(filename)
    data = csvdata[['end']]

    max, min = math.floor(csvdata[['end']].max()), math.floor(csvdata[['end']].min())
    x = []
    start = len(data) % LENGTH_OF_SEQUENCE
    for i in range(start, len(data) - LENGTH_OF_SEQUENCE):
        x.append(data.iloc[i:(i+LENGTH_OF_SEQUENCE)].as_matrix())
    x = np.asarray(x, 'float')
    result = normalize(max=max, min=min, data=x)
    resultoriginal = np.asarray(data[start + LENGTH_OF_SEQUENCE + 1:], 'float').tolist()
    return (resultoriginal, result.tolist(), max, min)



def loadModel():
    # ニューラルネットワークモデルを構築し、取得する
    in_out_neurons = 1
    hidden_neurons = 300
    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, LENGTH_OF_SEQUENCE, in_out_neurons),
                   return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def getSaveModelPath():
    return './model/keras-models.h5'

if __name__ == '__main__':
    X, Y = [], []

    learntargets = glob.glob('data/*.csv')
    for t in learntargets:
        _x, _y = loadCsvData(t)
        X.extend(_x)
        Y.extend(_y)

    Y = np.asarray(Y)
    X = np.asarray(X)

    traindatalength = math.floor(len(X) * 0.8)
    x_train, y_train = X[:traindatalength], Y[:traindatalength]
    x_test , y_test  = X[traindatalength:], Y[traindatalength:]

    # 学習
    model = loadModel()
    model.fit(X, Y, batch_size=100, nb_epoch=15, validation_split=0.05)
    model.save_weights(filepath=getSaveModelPath(), overwrite=True)

    # テスト結果表示
    predicted = model.predict(x_test)
    result = pd.DataFrame(predicted)
    result.columns = ['predict']
    result['actual'] = y_test
    result.plot()
    plt.show()

