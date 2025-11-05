import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import joblib
import os
import data_preprocess
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
def decision_Tree_regression(X, y):
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
    tree.fit(X, y)
    return tree

def setted_SVR(X, y):
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_rbf.fit(X, y)
    return svr_rbf


class time_sereis_regression():
    def __init__(self):
        self.df_list = []
        self.model = None

    def training(self, task="predict_one_day", save_path="", model_type="DT", n_lag=3):
        # task : predict_an_hour / predict_one_day / predict_three_days
        # model : dt(decision tree) / LR (Linear regression) / SVR (support vector regression)
        self.df_list = data_preprocess.time_series_preprocess("row_data")
        X, y = data_preprocess.time_series_preprocess(self.df_list, task, mode="training", n_lag=n_lag)
        if model_type == "DT":
            model = decision_Tree_regression(X, y)
        elif model_type == "SVR":
            model = setted_SVR(X, y)
        else:
            print("No such model type")
            return
        if save_path != "":
            joblib.dump(model, save_path)
            print("This model is complete saving.")
        else:
            print("This model didn't saved.")
        self.model = model

    def testing(self, task="predict_one_day", save_path=False, model_type="DT", n_lag=3):
        # model : dt(decision tree) / SVR (support vector regression)
        self.df_list = data_preprocess.load_csvs("row_data")
        X_training, y_training, X_testing, y_testing = data_preprocess.time_series_preprocess(self.df_list, task, mode="testing", n_lag=n_lag)
        if model_type == "DT":
            model = decision_Tree_regression(X_training, y_training)
        elif model_type == "SVR":
            model = setted_SVR(X_training, y_training)
        else:
            print("No such model type")
            return
        y_predict = model.predict(X_testing)
        print("MSE / mean of y:")
        print(np.mean((y_predict-y_testing)**2)**0.5/np.mean(y_testing))
        print("MAE / mean of y:")
        print(np.mean(np.abs(y_predict - y_testing))/np.mean(y_testing))
        print("Mean of y:")
        print(np.mean(y_testing))
        if task == "predict_one_day" or task == "predict_three_days":
            id = np.linspace(1, len(y_predict), len(y_predict))
            print(id.shape)
            print(y_testing.shape)
            plt.scatter(id, y_testing, marker='o', label='Ground Truth')
            plt.scatter(id, y_predict, marker='o', label='Predict')
            plt.legend()
            plt.show()
        if save_path:
            joblib.dump(model, save_path)
            print("This model is complete saving.")
        else:
            print("This model didn't saved.")
        self.model = model

    def predict(self, task = "predict_one_day", save_path = "", n_lag=3):
        if save_path != "":
            model = joblib.load(save_path)
            df_list = data_preprocess.time_series_preprocess("predict_needed_data")
            X = data_preprocess.time_series_preprocess(self.df_list, task, mode="predict", n_lag=n_lag)
            if task == "predict_one_day" or task == "predict_an_hour":
                result = model.predict(X)
                return result
            elif task == "predict_three_days":
                predict_list = []
                for i in range(3):
                    y_pred = model.predict(X)
                    predict_list.append(y_pred)
                    for j in range(n_lag-1):
                        X[:, j] = X[:, j+1]
                    X[:, -1] = y_pred
        else:
            print("Model hasn't been trained")
    def saving(self, save_path):
        if not self.model is None:
            joblib.dump(self.model, save_path)
            print("This model is complete saving.")


if __name__ == '__main__':
    model = time_sereis_regression()
    model.saving("test_save")
    model.testing(task="predict_one_day", model_type="DT")