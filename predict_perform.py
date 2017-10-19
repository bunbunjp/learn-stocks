import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loaddata import loadModel, getSaveModelPath, loadCsvDataXOnly, originalize

if __name__ == '__main__':
    model = loadModel()
    model.load_weights(getSaveModelPath())

    origin, X, maxval, minval = loadCsvDataXOnly('predict_data/7267-2017.csv')

    # テスト結果表示
    predicted = model.predict(X)
    predicted = originalize(min=minval, max=maxval, data=predicted)
    result = pd.DataFrame(predicted[:len(predicted) - 1])
    result.columns = ['predict']
    result['origin'] = np.asarray(origin)
    result.plot()

    move = predicted[len(predicted) - 1] - predicted[len(predicted) - 2]
    isUp = ((move) >= 0)
    if isUp:
        print('予想では前日より', move, '円 上昇する傾向です。')
    else :
        print('予想では前日より', move, '円 降下する傾向です。')
    print('前日の終値実値: ', origin[len(origin) - 1])
    print('今日の終値予想: ', predicted[len(predicted) - 1])

    plt.show()