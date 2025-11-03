import pandas as pd
import matplotlib.pyplot as plt
import data_preprocess
import numpy as np

def show_month_plot():
    july_data = data_preprocess.get_data("row_data/7月_用電量.csv")
    august_data = data_preprocess.get_data("row_data/8月_用電量.csv")
    july_data, x, y = data_preprocess.preprocess(july_data)
    august_data, x, y = data_preprocess.preprocess(august_data)
    mergy_show(july_data, august_data)


def mergy_show(df1, df2):
    df = pd.merge(df1, df2, on='day', how='outer', suffixes=('_A', '_B')).sort_values('day')

    plt.scatter(df['day'], df['_value_A'], marker='o', label='july')
    plt.scatter(df['day'], df['_value_B'], marker='s', label='august')
    plt.legend()
    plt.show()


show_month_plot()