import pandas as pd


def get_data(path):
    return pd.read_csv(path, header=1)
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