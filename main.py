import argparse
import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
import yaml

# Import custom modules
from data_provider import *
from model import *
from lightning_module import *
from statistical_model import *
from utils import *
import trainer

import hpo
import naive_prediction
import dfl
#import test

# Argument parser
parser = argparse.ArgumentParser(description='Train and predict using different models for stock data.')
parser.add_argument('--directory', type=str, default='../data/stock_data_prepro', help='Directory of stock data.')
parser.add_argument('--hpo', type=str, default ='n', help='hyperparameter optimization or not (y or n)')
parser.add_argument('--tsf', type=str, default ='n', help='TSF model training or not (y or n)')
parser.add_argument('--dfl', type=str, default ='n', help='End to End or not (y or n)')
parser.add_argument('--alpha', type=float, default =1, help='Hyperparameter between pred_loss and task loss')
parser.add_argument('--config', type=str, default ='configs.yml', help='Model configurations')
parser.add_argument('--tickers', type=str, nargs='+', default=['AAPL', 'TSLA', 'CVX','MSFT', 'JPM', 'AMZN', 'JNJ', 'LLY', 'XOM'], 
                    help='List of stock tickers.')
parser.add_argument('--weights', type=float, nargs='+', default=[(1, 0.01), (1, 0.02),(1, 0.1),(1, 0.02),(1, 0.1),(1, 0.5),(1, 0.1),(1, 0.3),(1, 0.1) ], 
                    help='List weight of prediction loss for each stock tickers.')
parser.add_argument('--prediction_length', type=int, default=78, help='Prediction Length.')
parser.add_argument('--context_length', type=int, default=78, help='Context Length.')
parser.add_argument('--wanted_interval', type=int, default=5, help='wanted time period.')
parser.add_argument('--time_length', type=int, default=390, help='One day time length according to the prediction interval')
parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio.')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio.')
parser.add_argument('--max_epochs', type=int, default=30, help='Max epoch.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
parser.add_argument('--gpu', type=bool, default =True, help='GPU')
parser.add_argument('--opt_method', type=str, default ='delay', help='Determine the optimizing methods (delay, index ,double)')
parser.add_argument('--model_dir', type=str, help='directory for loading model parameters (must match with the configuration)')
parser.add_argument('--save_dir', type=str, default ='./saved', help='directory for saving model parameters')
parser.add_argument('--learning_rate', type=float, default =0.0005, help='directory for saving model parameters')
parser.add_argument('--test', type=str, default ='n', help='directory for saving model parameters')
args = parser.parse_args()


# Get logger
args.accelerator = "cuda" if args.gpu else 'cpu'

# Load configurations from the YAML file
configurations = config.load_config(args.config)

# Loop through the configurations and instantiate the models
models = []
results = []
# hyperparameter optimization for prediction
if args.hpo != 'n' :
    configurations = hpo.optimize_hyperparameters(args.save_dir, configurations, train_loader, val_loader, args.accelerator, args.max_epochs)

# Train and test each model
# training case : Train and test each model

def assert_weights_tickers_is_valid(tickers, weights):
    assert len(weights) == len(tickers), "Both elements must have the same length."
    
    for weight in weights : 
        assert isinstance(weight, tuple), "Input must be a tuple."
        assert len(weight) == 2, "Tuple must have exactly 2 elements."
        assert all(isinstance(x, (int, float)) for x in weight), "Both elements must be int or float."
assert_weights_tickers_is_valid(args.tickers, args.weights)
                                                                 
for ticker, weight in zip(args.tickers, args.weights) : 
    # data preprocessing
    scaler, train_loader, val_loader, test_loader = initialize_data_module(args.directory, ticker, args.time_length, args.prediction_length, args.context_length, args.wanted_interval, args.train_ratio, args.val_ratio, args.batch_size)
    
    # Create a directory for saving results
    save_dir = args.save_dir + '_' +ticker
    print(args.save_dir, ticker)
    os.makedirs(save_dir, exist_ok=True)

    # Predict, the optimize methods, time-series forecasting model training 
    if args.tsf !='n' :
        models, _ = naive_prediction.training_prediction_models(save_dir, configurations, train_loader, val_loader, test_loader, scaler, args.accelerator, args.max_epochs, ticker)
      
    # load model parameters
    if args.model_dir and not (args.test=='yes'):
        for model_name, config in configurations.items():
            # Load model parameters from file
            model, module_type = load_model_parameters(model_name, config, args.model_dir+'_'+ticker+'/'+f'{model_name}_params.pth')    
            models.append(model)
            #pred, target = load_model_prediction(model, test_loader)
            #results.append((pred, target))

    #Decision-Focused methods
    if args.dfl != 'n' :
        models, results = dfl.training_dfl_models (save_dir,  models, args , configurations, ticker, weight)

    """
    if args.test != 'n' and args.model_dir  :
        for model_name, config in configurations.items():
            # Load model parameters from file
            model, module_type = load_model_parameters(model_name, config, args.model_dir+'_'+ticker+'/'+f'DFL_{model_name}_params.pth')    
            models.append(model)

        test.test_models(args.save_dir,  models, args , configurations, step_size)
    """

configuration_names = list(configurations.keys())
configuration_names.append('Vwap')
configuration_names.append('Total')
# MA

#models.append('Vwap')
#results.append(  training_moving_average(args) )
#models.append('Total')
#results.append(  training_total_average(args) )
                                                                
                                                                 
                                                                 
