import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import joblib
import os
import data_preprocess
from datetime import datetime
import matplotlib.pyplot as plt
def decision_Tree_regression(X, y):
    tree = DecisionTreeRegressor(max_depth=5)
    tree.fit(X, y)
    return tree

def setted_SVR(X, y):
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_rbf.fit(X, y)
    return svr_rbf
class decision_tree_method():
    def __init__(self):
        self.df_list = []
    def training(self, task = "predict_one_day", save_path = "", model_type = "DT"):
        # model : dt(decision tree) / LR (Linear regression) / SVR (support vector regression)
        self.df_list = data_preprocess.load_csvs("row_data")
        X, y = data_preprocess.training_preprocess(self.df_list, task)
        if model_type == "DT":
            model = decision_Tree_regression(X, y)
        elif model_type == "SVR":
            model = setted_SVR(X, y)
        else:
            print("No such model type")
            return
        if save_path != "":
            joblib.dump(model, save_path)
        else:
            print("This model didn't saved.")

    def testing(self, task="predict_one_day", save_path=False, model_type = "DT"):
        # model : dt(decision tree) / LR (Linear regression) / SVR (support vector regression)
        self.df_list = data_preprocess.load_csvs("row_data")
        X_training, y_training, X_testing, y_testing = data_preprocess.testing_preprocess(self.df_list, task)
        if model_type == "DT":
            model = decision_Tree_regression(X_training, y_training)
        elif model_type == "SVR":
            model = setted_SVR(X_training, y_training)
        else:
            print("No such model type")
            return
        y_predict = model.predict(X_testing)
        plt.scatter(X_testing[:, 1], y_testing, marker='o', label='Ground Truth')
        plt.scatter(X_testing[:, 1], y_predict, marker='o', label='Predict')
        plt.legend()
        plt.show()
        if save_path:
            joblib.dump(model, save_path)
            print("This model is complete saving.")
        else:
            print("This model didn't saved.")
    def predict(self, task = "predict_one_day", save_path = ""):
        #model : dt(decision tree) / LR (Linear regression) / SVR (support vector regression)
        if save_path != "":
            model = joblib.load(save_path)
            current_dateTime = datetime.now()
            if task == "predict_one_day":
                today = np.asarray([current_dateTime.month, current_dateTime.day]).reshape(1, -1)
                result = model.predict(today)
                return result
        else:
            print("Model hasn't been trained")


if __name__ == '__main__':
    model = decision_tree_method()
    model.training(save_path="test_SVR_save", model_type="SVR")
    print(model.predict(save_path="test_SVR_save"))