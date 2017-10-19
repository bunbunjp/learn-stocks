import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loaddata import loadModel, getSaveModelPath, loadCsvData

if __name__ == '__main__':
    # 学習データの読み込み
    model = loadModel()
    model.load_weights(getSaveModelPath())

    X, Y = loadCsvData('data/6501-2016-2015.csv')

    # テスト結果表示
    predicted = model.predict(X)
    result = pd.DataFrame(predicted)
    result.columns = ['predict']
    result['actual'] = np.asarray(Y)
    result.plot()
    plt.show()
