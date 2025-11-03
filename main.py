# This is a sample Python script.
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


july_data = pd.read_csv("row_data/7月_用電量.csv", header=1)
august_data = pd.read_csv("row_data/8月_用電量.csv", header=1)
def split_time(table):
    split_cols = table['_time'].str.split('T', expand=True)
    split_cols.columns = ['date', 'time']
    split_cols_date = split_cols['date'].str.split('-', expand=True)
    split_cols_date.columns = ['year', 'month', 'day']
    split_cols_time = split_cols['time'].str.split(':', expand=True)
    split_cols_time.columns = ['hour', 'minute', 'second']
    table = pd.concat([split_cols_date, split_cols_time, july_data.drop(columns=['_time'])], axis=1)
    return table

def group_by_day(df):
    result = df.groupby([ 'month', 'day']).agg({
        'year': 'first',
        'month': 'first',
        'day': 'first',
        '_value': 'sum'
    })
    X = result.iloc[:, 1:3].values
    y = result.iloc[:, 3].values
    return result, X, y

def decision_Tree_regression(X, y):
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, y)
    return tree

july_data = pd.concat([july_data, august_data])
july_data = split_time(july_data)
july_data, X, y = group_by_day(july_data)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
tree = DecisionTreeRegressor(max_depth=2)
tree.fit(X, y)
y_pred = tree.predict(X)

plt.scatter(X[:,1], y, color='blue', label='data')
plt.plot(X[:,1], y_pred, color='red', label='SVR (RBF kernel)')
plt.legend()
plt.show()


if __name__ == '__main__':
    '''
    如果尚未建立模型或是要求訓練模型:
    從row data 提取訓練資料並訓練、儲存模型。

    提取當前時間，預測當日、未來三日、未來一小時之用電量。
    :return: 預測結果
    '''
    pass