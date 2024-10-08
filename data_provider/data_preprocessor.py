# data_preprocessor.py
import pandas as pd
import numpy as np

def preprocess_date_time(df):
    """
    Preprocess date-time index of DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
    Returns:
        pd.DataFrame: DataFrame with date-time index preprocessed.
    """
    date_time_idx = pd.date_range('2019-12-01', periods=df.shape[0], freq='T')
    df['tmp_date_time'] = date_time_idx
    df['volume'] = np.log(df['volume'])
    df = df.drop(['day', 'one_min'], axis=1)
    df['id'] = 0
    return df

def preprocess_sliding_window(df, time_length, iw, ow) :
    """
    Preprocess sliding window for sequential time series forecasting
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        time_length (int) : one day period
        iw (int) : Context Length
        ow (int) : prediction Length
    Returns:
        X : input data sequence torch
        Y : target sequence torch
    """
    # Except for things that don't break on a daily interval.
    iw_except = max( iw, time_length)
    ow_except = max( ow, time_length)
    
    # except first two days for input seqence, except last one day because there is no prediction after that day (the prediction predict beyond the day)
    X = np.array([df[i - iw_except:i] for i in range(iw_except, df.shape[0] - ow_except)])
    Y = np.array([df[i:i + ow_except] for i in range(iw_except, df.shape[0] - ow_except)])
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=2)
        Y = np.expand_dims(Y, axis=2)

    return X, Y

def preprocess_one_day(df, time_length, iw, ow) :
    """
    Preprocess sliding window for sequential time series forecasting
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        time_length (int) : one day period
        iw (int) : Context Length
        ow (int) : prediction Length
    Returns:
        X : input data sequence torch
        Y : target sequence torch
    """
    # Except for things that don't break on a daily interval.
    

    return X, Y

def preprocess_interval (df, time_length, wanted) :
    """
    Recreate the dataframe with wanted time interval dataframe 
    Args:
        df (pd.DataFrame): DataFrame containing stock data
        time_length (int) : given dataframe with one day period
        wanted (int) : wanted time interval ( e.g 15min, 5min interval and so on)
    Returns:
        summed_df (pd.DataFrame): DataFrame with summed
    """
    # total minute per day
    minute_time_length = 390
    # not divided into a day
    if minute_time_length%wanted != 0 : assert print("Not appropriate Time")
    # the length per day after switch
    modified_time_length = (minute_time_length // wanted)
    concatenate_len = minute_time_length // modified_time_length # (390//15) => 3
    print( "1 min stock data is changed into ", wanted, "min stock data: ", modified_time_length, " per day")
    
    # Reshape the DataFrame into groups of three rows and sum each group
    summed_df = df.groupby(df.index // concatenate_len).sum()
    return np.log(summed_df), modified_time_length
        
    
    