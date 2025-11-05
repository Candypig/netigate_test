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
    #X = result.iloc[:, 1:3].values
    #y = result.iloc[:, 3].values
    return result


def group_by_hour(df):
    result = df.groupby(['month', 'day', 'hour'], as_index=False).agg({
        'year': 'first',
        'month': 'first',
        'day': 'first',
        'hour': 'first',
        '_value': 'sum'
    })
    return result


def preprocess(df, type="month"):
    pre_df = split_time(df)
    if type == "month":
        pre_df = group_by_day(pre_df)
        return pre_df
    return pre_df


def training_preprocess(dflist, mode="predict_one_day"):
    # mode : predict_an_hour / predict_one_day / predict_three_days
    for i, df in enumerate(dflist):
        dflist[i] = split_time(df)
    if mode == "predict_one_day" or "predict_three_days":
        for i, df in enumerate(dflist):
            dflist[i] = group_by_day(df)
        stack_list = pd.concat(dflist, ignore_index=True)
        return stack_list[['month', 'day']], stack_list['_value']
    elif mode == "predict_an_hour" or "predict_three_days":
        for i, df in enumerate(dflist):
            dflist[i] = group_by_hour(df)
        stack_list = pd.concat(dflist, ignore_index=True)
        return stack_list[['month', 'day', 'hour']], stack_list['_value']


def testing_preprocess(dflist, mode="predict_one_day"):
    for i, df in enumerate(dflist):
        dflist[i] = split_time(df)
    if mode == "predict_one_day":
        for i, df in enumerate(dflist):
            dflist[i] = group_by_day(df)
        stack_list = pd.concat(dflist[:-1], ignore_index=True)

        X_training = stack_list[['month', 'day']].values
        y_training = stack_list['_value'].values
        X_testing = dflist[-1][['month', 'day']].values
        y_testing = dflist[-1]['_value'].values
        return X_training, y_training, X_testing, y_testing
    if mode == "predict_an_hour":
        for i, df in enumerate(dflist):
            dflist[i] = group_by_hour(df)
        stack_list = pd.concat(dflist[:-1], ignore_index=True)

        X_training = stack_list[['month', 'day', 'hour']].values
        y_training = stack_list['_value'].values
        X_testing = dflist[-1][['month', 'day', 'hour']].values
        y_testing = dflist[-1]['_value'].values
        return X_training, y_training, X_testing, y_testing


def time_series_preprocess(dflist, task="predict_one_day", mode="training", n_lag=3):
    for i, df in enumerate(dflist):
        dflist[i] = split_time(df)
    stack_list = pd.concat(dflist, ignore_index=True)
    if task == "predict_one_day" or task == "predict_three_days":
        df = group_by_day(stack_list)
        df.sort_values(['month', 'day'])
        X, y = set_n_lag_df(df, n_lag)
        if mode == "training":
            return X, y
        elif mode == "testing":
            train_size = int(len(df) * 0.8)
            X_training, X_testing = X[:train_size], X[train_size:]
            y_training, y_testing = y[:train_size], y[train_size:]
            return X_training, y_training, X_testing, y_testing
        elif mode == "predict":
            return X.tail(1)
    elif task == "predict_an_hour":
        df = group_by_hour(stack_list)
        df.sort_values(['month', 'day', 'hour'])
        X, y = set_n_lag_df(df, n_lag)
        if mode == "training":
            return X, y
        elif mode == "testing":
            train_size = int(len(df) * 0.8)
            X_training, X_testing = X.iloc[:train_size], X.iloc[train_size:]
            y_training, y_testing = y.iloc[:train_size], y.iloc[train_size:]
            return X_training, y_training, X_testing, y_testing
        elif mode == "predict":
            return X.tail(1).values


def set_n_lag_df(df, n_lag=3):
    for i in range(1, n_lag + 1):
        df[f'lag_{i}'] = df['_value'].shift(i)
    df = df.dropna()
    X = df[[f'lag_{i}' for i in range(1, n_lag + 1)]].values
    y = df['_value'].values
    return X, y


if __name__ == '__main__':
    # small function test
    dflist = load_csvs("row_data")
    df, _, _ = time_series_preprocess(dflist)
    print(df.head(5))