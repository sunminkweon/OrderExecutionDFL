import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from data_provider import data_reader, data_preprocessor, data_splitter
from sklearn.preprocessing import StandardScaler
from utils import save_to_csv

class TotalAverage():
    def __init__(self,
                 directory: str = "stock_data_prepro",
                 ticker: str = 'AAPL_min',
                 save_dir : str = './saved',
                 time_length: int = 390,
                 wanted_interval: int = 1,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 prediction_length : int = 78
                 ):
        super().__init__()
        self.directory = directory
        self.ticker = ticker
        self.save_dir = save_dir
        self.time_length = time_length
        self.wanted_interval = wanted_interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.pred_len = prediction_length
    
    def setup(self, stage=None):
        # Load and preprocess your data
        df = data_reader.read_stock_data(self.directory, self.ticker)
        df_vol = df['volume']
        df_vol, self.time_length  = data_preprocessor.preprocess_interval(df_vol , self.time_length, self.wanted_interval)

        total_day_num = df_vol.shape[0] // self.time_length
        training_day_num = (int) (total_day_num * self.train_ratio)
        validation_day_num = (int) (total_day_num * self.val_ratio)
        test_day_num = total_day_num - training_day_num - validation_day_num
        
        # when the prediction is not one day cases
        self.train_df = df_vol.head(int((training_day_num+validation_day_num)*self.time_length))
        test_df = df_vol.tail(int((test_day_num)* self.time_length))

        # Preprocess data into input-output pairs
        # train_X : all prediction times ( e.g 5days pred : 5*78 x average window x pred_len ) 
        self.train_X, self.target = data_preprocessor.preprocess_sliding_window(test_df, self.time_length,  iw=self.time_length , ow=self.pred_len )
        
    # prediction based on past observastions    
    def predict(self):
        num_days =  self.target.shape[0] // self.time_length
        self.prediction_start = np.mean( np.array(self.train_df).reshape(-1,self.time_length) , axis=0) #np.tile( np.mean( np.array(self.train_df).reshape(-1,self.time_length) , axis=0) , num_days).reshape(-1,self.time_length)
        self.prediction = np.zeros ( (num_days*self.time_length, self.time_length))
        
        # Rotate each column
        for num_day_idx in range(num_days) :
            for t in range(self.time_length) :
                self.prediction[num_day_idx*self.time_length+t,:] = np.concatenate((self.prediction_start[t:], self.prediction_start[:t]), axis=0)        
        self.prediction = np.concatenate([self.prediction] *(int) (self.pred_len/ self.time_length) , axis=1)
        
    def return_prediction(self):
        return self.prediction, self.target
        
    def save_prediction(self) :
        save_to_csv( self.prediction.squeeze().reshape(-1,1), "Naive_total" , self.save_dir)
        
def training_total_average(args):
    total_average_model = TotalAverage(
        directory=args.directory,
        ticker=args.ticker,
        save_dir= args.save_dir,
        time_length=args.time_length, 
        wanted_interval = args.wanted_interval,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        prediction_length = args.prediction_length
    )
    total_average_model.setup()
    
    # Make predictions using the MovingAverage model
    total_average_model.predict()
    total_average_prediction, total_average_target = total_average_model.return_prediction()
    total_average_model.save_prediction()

    return total_average_prediction, total_average_target
