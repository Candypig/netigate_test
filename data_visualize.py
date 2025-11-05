import pandas as pd
import matplotlib.pyplot as plt
import data_preprocess
from functools import reduce
import simple_regression
import time_series_method

def show_month_plot():
    july_data = data_preprocess.get_data("row_data/7月_用電量.csv")
    august_data = data_preprocess.get_data("row_data/8月_用電量.csv")
    sep_data = data_preprocess.get_data("row_data/9月_用電量.csv")
    oct_data = data_preprocess.get_data("row_data/10月_用電量.csv")
    july_data = data_preprocess.preprocess(july_data)
    august_data = data_preprocess.preprocess(august_data)
    sep_data = data_preprocess.preprocess(sep_data)
    oct_data = data_preprocess.preprocess(oct_data)
    mergy_show([july_data, august_data, sep_data, oct_data])


def mergy_show(dflist):
    #df = pd.merge(df1, df2, on='day', how='outer', suffixes=('_A', '_B')).sort_values('day')
    df = merge_dfs(dflist)


    plt.scatter(df['day'], df['df1__value'], marker='o', label='july')
    plt.scatter(df['day'], df['df2__value'], marker='o', label='august')
    plt.scatter(df['day'], df['df3__value'], marker='o', label='sep')
    plt.scatter(df['day'], df['df4__value'], marker='o', label='oct')
    plt.legend()
    plt.show()



def merge_dfs(df_list, index_col='day', prefix='df'):

    renamed_list = []
    for i, df in enumerate(df_list):
        new_cols = {col: f"{prefix}{i + 1}_{col}" for col in df.columns if col != index_col}
        renamed_list.append(df.rename(columns=new_cols))

    # 使用 merge 迭代合併
    merged = reduce(lambda left, right: pd.merge(left, right, on=index_col, how='outer'), renamed_list)

    return merged

if __name__ == '__main__':
    show_month_plot()
    s_model = simple_regression.simple_regression()
    s_model.testing(task="predict_one_day", model_type="DT")
    s_model.testing(task="predict_one_day", model_type="SVR")
    t_model = time_series_method.time_sereis_regression()
    t_model.testing(task="predict_one_day", model_type="LR")
    t_model.testing(task="predict_one_day", model_type="DT")
    t_model.testing(task="predict_one_day", model_type="SVR")
