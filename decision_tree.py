import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib
import os
import data_preprocess
from datetime import datetime
import matplotlib.pyplot as plt
def decision_Tree_regression(X, y, save_path):
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, y)
    joblib.dump(tree, save_path)
    return tree

def training_setting(mode = "testing", task = "predict_one_day", save_path = False):
    # mode : training / train_and_test / predict
    # mode : predict_an_hour / predict_one_day / predict_three_days
    df_list = data_preprocess.load_csvs("row_data")
    if mode == "training":
        training(task, save_path)

def training(task = "predict_one_day", save_path = False):
    X, y = data_preprocess.training_preprocess(df_list, task)
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, y)
    if save_path:
        joblib.dump(tree, save_path)
    else:
        print("This model didn't save")



class decision_tree_method():
    def __init__(self):
        self.df_list = []
    def training(self, task = "predict_one_day", save_path = False):
        self.df_list = data_preprocess.load_csvs("row_data")
        X, y = data_preprocess.training_preprocess(self.df_list, task)
        tree = DecisionTreeRegressor(max_depth=2)
        tree.fit(X, y)
        if save_path:
            joblib.dump(tree, save_path)
        else:
            print("This model didn't saved.")

    def testing(self, task="predict_one_day", save_path=False):
        self.df_list = data_preprocess.load_csvs("row_data")
        X_training, y_training, X_testing, y_testing = data_preprocess.testing_preprocess(self.df_list, task)
        tree = DecisionTreeRegressor(max_depth=5)
        tree.fit(X_training, y_training)
        y_predict = tree.predict(X_testing)
        plt.scatter(X_testing[:, 1], y_testing, marker='o', label='Ground Truth')
        plt.scatter(X_testing[:, 1], y_predict, marker='o', label='Predict')
        plt.legend()
        plt.show()
        if save_path:
            joblib.dump(tree, save_path)
            print("This model is complete saving.")
        else:
            print("This model didn't saved.")
    def predict(self, task = "predict_one_day", save_path = False):
        if save_path:
            tree = joblib.load(save_path)
            current_dateTime = datetime.now()
            if task == "predict_one_day":
                today = np.asarray([current_dateTime.month, current_dateTime.day]).reshape(1, -1)
                result = tree.predict(today)
                return result
        else:
            print("Model hasn't been trained")

model = decision_tree_method()
model.training(save_path="test_save")
print(model.predict(save_path="test_save"))