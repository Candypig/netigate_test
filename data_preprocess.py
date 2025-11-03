import pandas as pd
import os

def get_data(path):
    return pd.read_csv(path, header=1)

def load_csvs(folder_path):
    df_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.csv')):
            df = pd.read_csv(os.path.join(folder_path, filename), header=1)
            df_list.append(df)
    return df_list
def split_time(table):
    split_cols = table['_time'].str.split('T', expand=True)
    split_cols.columns = ['date', 'time']
    split_cols_date = split_cols['date'].str.split('-', expand=True)
    split_cols_date.columns = ['year', 'month', 'day']
    split_cols_time = split_cols['time'].str.split(':', expand=True)
    split_cols_time.columns = ['hour', 'minute', 'second']
    table = pd.concat([split_cols_date, split_cols_time, table.drop(columns=['_time'])], axis=1)
    return table
def group_by_day(df):
    result = df.groupby([ 'month', 'day'], as_index=False).agg({
        'year': 'first',
        'month': 'first',
        'day': 'first',
        '_value': 'sum'
    })
    X = result.iloc[:, 1:3].values
    y = result.iloc[:, 3].values
    return result, X, y

def preprocess(df, type = "month"):
    pre_df = split_time(df)
    if type == "month":
        pre_df, X, y = group_by_day(pre_df)
        return pre_df, X, y
    return pre_df

def training_preprocess(dflist, mode = "predict_one_day"):
    # mode : predict_an_hour / predict_one_day / predict_three_days
    for i, df in enumerate(dflist):
        dflist[i] = split_time(df)
    if mode == "predict_one_day":
        for i, df in enumerate(dflist):
            dflist[i], _, _ = group_by_day(df)
        stack_list = pd.concat(dflist, ignore_index=True)
        return stack_list[['month', 'day']], stack_list['_value']

def testing_preprocess(dflist, mode = "predict_one_day"):
    for i, df in enumerate(dflist):
        dflist[i] = split_time(df)
    if mode == "predict_one_day":
        for i, df in enumerate(dflist):
            dflist[i], _, _ = group_by_day(df)
        stack_list = pd.concat(dflist[:-1], ignore_index=True)
        print(stack_list.head(5))
        X_training = stack_list[['month', 'day']].values
        y_training = stack_list['_value'].values
        X_testing = dflist[-1][['month', 'day']].values
        y_testing = dflist[-1]['_value'].values
        return X_training, y_training, X_testing, y_testing