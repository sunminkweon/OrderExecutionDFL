import pandas as pd
import random

def split_data_day(df, time_length, train_ratio, val_ratio):
    """
    Split data into training, validation, and test sets.
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        time_length (int): Length of time sequence.
        train_ratio (float) : ratio for training.
        val_ratio (float) : ratio for validation set.
    Returns:
        tuple: Training, validation, and test DataFrames.
    """
    
    total_day_num = df.shape[0] // time_length 
    training_day_num = (int) (total_day_num * train_ratio)
    validation_day_num = (int) (total_day_num * val_ratio)
    test_day_num = total_day_num - training_day_num - validation_day_num
    
    training_df = df.head(int(training_day_num * time_length))
    validation_df = df.iloc[int(training_day_num * time_length):int((training_day_num + validation_day_num) * time_length)]
    
    test_df = df.iloc[int((training_day_num + validation_day_num) * time_length):]
    
    return training_df, validation_df, test_df
    
def split_data_random(df, time_length, train_ratio, val_ratio):
    """
    Split data into training, validation, and test sets.
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
        time_length (int): Length of time sequence.
        train_ratio (float) : ratio for training.
        val_ratio (float) : ratio for validation set.
    Returns:
        tuple: Training, validation, and test DataFrames.
    """
    total_day_num = df.shape[0] // time_length 
    training_day_num = (int) (total_day_num * train_ratio)
    validation_day_num = (int) (total_day_num * val_ratio)
    test_day_num = total_day_num - training_day_num - validation_day_num

    #random sampling for train and validation within days
    
    train_and_validation_df =  df.head(int(training_day_num+ validation_day_num) * time_length)
    
    validation_df = pd.concat([train_and_validation_df.iloc[index*time_length:index*time_length + time_length] for index in random.sample(range(0,training_day_num+validation_day_num),validation_day_num)])
    training_df = train_and_validation_df.drop(validation_df.index) 
    
    #training_df = df.head(int(training_day_num * time_length))
    #validation_df = df.iloc[int(training_day_num * time_length):int((training_day_num + validation_day_num) * time_length)]
    
    test_df = df.iloc[int((training_day_num + validation_day_num) * time_length):]
    print(total_day_num, training_day_num, validation_day_num, test_day_num, train_ratio, val_ratio)
    
    return training_df, validation_df, test_df