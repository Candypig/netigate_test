# This is a sample Python script.
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import simple_regression
import time_series_method




if __name__ == '__main__':
    '''
    如果尚未建立模型或是要求訓練模型:
    從row data 提取訓練資料並訓練、儲存模型。

    提取當前時間，預測當日、未來三日、未來一小時之用電量。
    :return: 預測結果
    '''
    '''
    model_type : DT(decision tree) / SVR (support vector regression)
    task : predict_an_hour / predict_one_day / predict_three_days
    '''
    # test
    s_model = simple_regression.simple_regression()
    s_model.testing(task="predict_one_day", model_type="DT")
    t_model = time_series_method.time_sereis_regression()
    # time series can set n_lag
    t_model.testing(task="predict_an_hour", model_type="DT", n_lag=3)
    # train
    s_model = simple_regression.simple_regression()
    s_model.training(task="predict_one_day", model_type="DT", save_path="s_model")
    t_model = time_series_method.time_sereis_regression()
    t_model.training(task="predict_an_hour", model_type="DT", save_path="t_model", n_lag=3)
    # predict
    # Use model hasn't trained, need to set the saving_path of model train and save before.
    print(s_model.predict(save_path="s_model"))
    print(t_model.predict(save_path="t_model"))
