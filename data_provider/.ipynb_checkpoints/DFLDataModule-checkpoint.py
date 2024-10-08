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
import sklearn.preprocessing 

class EndtoEndDataModule(pl.LightningDataModule):
    def __init__(self, args,  step):
        super().__init__()

        self.directory = args.directory
        self.ticker = args.ticker
        self.time_length = args.time_length
        self.prediction_length = args.prediction_length
        self.context_length = args.context_length
        self.wanted_interval = args.wanted_interval
        self.device = args.accelerator
        self.step =step
        
        self.train_ratio = args.train_ratio
        self.val_ratio = args.val_ratio
        self.batch_size = args.batch_size
        self.scaler = sklearn.preprocessing.StandardScaler() #Scaler()
        
    def setup(self, stage=None):
        
        # Load and preprocess your data
        df = data_reader.read_stock_data(self.directory, self.ticker)
        
        # Log transformation
        df_vol = (df['volume'])
        df_vol, self.time_length = data_preprocessor.preprocess_interval(df_vol , self.time_length, self.wanted_interval)
       
        train_df, validation_df, test_df = data_splitter.split_data_day(df_vol, self.time_length, self.train_ratio, self.val_ratio)

        train_np = np.array(train_df).reshape(-1,1)
        validation_np = np.array( validation_df).reshape(-1,1)
        test_np = np.array(test_df).reshape(-1,1)

        train_df = self.scaler.fit_transform(np.array(train_df).reshape(-1,1))
        validation_df = self.scaler.transform(np.array(validation_df).reshape(-1,1))
        test_df = self.scaler.transform(np.array(test_df).reshape(-1,1))

        # Preprocess data into input-output pairs
        train_X, train_Y = data_preprocessor.preprocess_sliding_window(train_df, self.time_length, self.context_length, self.prediction_length)
        val_X, val_Y = data_preprocessor.preprocess_sliding_window(validation_df, self.time_length, self.context_length, self.prediction_length)
        test_X, test_Y = data_preprocessor.preprocess_sliding_window(test_df, self.time_length, self.context_length, self.prediction_length)

  
        train_X = train_X.reshape(-1,self.time_length, self.time_length)[:,0::self.step]
        val_X = val_X.reshape(-1,self.time_length, self.time_length)[:,0::self.step]
        test_X = test_X.reshape(-1,self.time_length, self.time_length)[:,0::self.step]
        
        train_Y = train_Y.reshape(-1,self.time_length, self.time_length)[:,0::self.step]
        val_Y = val_Y.reshape(-1,self.time_length, self.time_length)[:,0::self.step]
        test_Y = test_Y.reshape(-1,self.time_length, self.time_length)[:,0::self.step]
        
        # Convert to PyTorch tensors
        self.train_dataset = TensorDataset(torch.Tensor(train_X), torch.Tensor(train_Y) )
        self.val_dataset = TensorDataset(torch.Tensor(val_X), torch.Tensor(val_Y) )
        self.test_dataset = TensorDataset(torch.Tensor(test_X), torch.Tensor(test_Y))

    def return_scaler(self):
        return self.scaler
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)

def initialize_data_module(args, step):
    
    data_module = EndtoEndDataModule(args, step)
    
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    scaler = data_module.return_scaler()
    return train_loader,val_loader,test_loader, scaler
    