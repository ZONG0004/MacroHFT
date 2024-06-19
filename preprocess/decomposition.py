import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression

def smooth_data(data):
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    return filtfilt(b, a, data)

def get_slope(smoothed_data):
    X = np.arange(len(smoothed_data)).reshape(-1, 1)
    model = LinearRegression().fit(X, smoothed_data)
    return model.coef_[0]

def get_slope_window(window):
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    y = filtfilt(b, a, window.values)
    X = np.arange(len(y)).reshape(-1, 1)   
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

def chunk(df_train, df_val, df_test):
    chunk_size = 4320
    for i in range(int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_train[start:end].reset_index(drop=True)
        df_chunk.to_feather('./data/ETHUSDT/train/df_{}.feather'.format(i))

    for i in range(int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_val[start:end].reset_index(drop=True)
        df_chunk.to_feather('./data/ETHUSDT/val/df_{}.feather'.format(i))

    for i in range(int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_test[start:end].reset_index(drop=True)
        df_chunk.to_feather('./data/ETHUSDT/test/df_{}.feather'.format(i))

def label_slope(df_train, df_val, df_test):
    chunk_size = 4320
    slopes_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_train.append(slope)

    slopes_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_val.append(slope)

    slopes_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_test.append(slope)

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    slope_labels_train, bins = pd.qcut(slopes_train, q=quantiles, retbins=True, labels=False)

    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(slope_labels_train):
        train_indices[label].append(index)
    with open('./data/ETHUSDT/train/slope_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    bins[0] = -100
    bins[-1] = 100
    slope_labels_val = pd.cut(slopes_val, bins=bins, labels=False, include_lowest=True)
    slope_labels_val = [1 if element == 0 else element for element in slope_labels_val]
    slope_labels_val = [3 if element == 4 else element for element in slope_labels_val]
    slope_labels_test = pd.cut(slopes_test, bins=bins, labels=False, include_lowest=True)
    slope_labels_test = [1 if element == 0 else element for element in slope_labels_test]
    slope_labels_test = [3 if element == 4 else element for element in slope_labels_test]

    for index, label in enumerate(slope_labels_val):
        val_indices[label].append(index)
    with open('./data/ETHUSDT/val/slope_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(slope_labels_test):
        test_indices[label].append(index)
    with open('./data/ETHUSDT/test/slope_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)

def label_volatility(df_train, df_val, df_test):
    chunk_size = 4320
    volatilities_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_train.append(volatility)

    volatilities_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_val.append(volatility)
    
    volatilities_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_test.append(volatility)

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    vol_labels_train, bins = pd.qcut(volatilities_train, q=quantiles, retbins=True, labels=False)

    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(vol_labels_train):
        train_indices[label].append(index)
    with open('./data/ETHUSDT/train/vol_labels.pkl', 'wb') as file:
        pickle.dump(train_indices, file)

    bins[0] = 0
    bins[-1] = 1
    vol_labels_val = pd.cut(volatilities_val, bins=bins, labels=False, include_lowest=True)
    vol_labels_val = [1 if element == 0 else element for element in vol_labels_val]
    vol_labels_val = [3 if element == 4 else element for element in vol_labels_val]
    vol_labels_test = pd.cut(volatilities_test, bins=bins, labels=False, include_lowest=True)
    vol_labels_test = [1 if element == 0 else element for element in vol_labels_test]
    vol_labels_test = [3 if element == 4 else element for element in vol_labels_test]

    for index, label in enumerate(vol_labels_val):
        val_indices[label].append(index)
    with open('./data/ETHUSDT/val/vol_labels.pkl', 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(vol_labels_test):
        test_indices[label].append(index)
    with open('./data/ETHUSDT/test/vol_labels.pkl', 'wb') as file:
        pickle.dump(test_indices, file)

def label_whole(df):
    window_size_list = [360]
    for i in range(len(window_size_list)):
        window_size = window_size_list[i]
        df['slope_{}'.format(window_size)] = df['close'].rolling(window=window_size).apply(get_slope_window)
        df['return'] = df['close'].pct_change().fillna(0)
        df['vol_{}'.format(window_size)] = df['return'].rolling(window=window_size).std()
    return df

if __name__ == "__main__":
    df_train = pd.read_feather('./data/ETHUSDT/df_train.feather')
    df_val = pd.read_feather('./data/ETHUSDT/df_val.feather')
    df_test = pd.read_feather('./data/ETHUSDT/df_test.feather')

    os.makedirs('./data/ETHUSDT/train', exist_ok=True)
    os.makedirs('./data/ETHUSDT/val', exist_ok=True)
    os.makedirs('./data/ETHUSDT/test', exist_ok=True)
    os.makedirs('./data/ETHUSDT/whole', exist_ok=True)

    chunk(df_train, df_val, df_test)
    label_slope(df_train, df_val, df_test)
    label_volatility(df_train, df_val, df_test)

    df_train = label_whole(df_train).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_val = label_whole(df_val).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_test = label_whole(df_test).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)

    df_train.to_feather('./data/ETHUSDT/whole/train.feather')
    df_val.to_feather('./data/ETHUSDT/whole/val.feather')
    df_test.to_feather('./data/ETHUSDT/whole/test.feather')


    