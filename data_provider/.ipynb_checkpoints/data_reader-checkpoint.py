import os
import pandas as pd

def read_stock_data(directory, ticker):
    """
    Read stock data files from a directory for a specific ticker.
    Args:
        directory (str): Directory containing stock data files.
        ticker (str): Ticker symbol.
    Returns:
        pd.DataFrame: Concatenated DataFrame of stock data for the ticker.
    """
    df = pd.DataFrame()
    fns = sorted(os.listdir(directory))
    
    for fn in fns:
        if fn.find(ticker) != -1 :
            df_tmp = pd.read_csv(os.path.join(directory, fn))
            df = pd.concat([df, df_tmp])
    return df