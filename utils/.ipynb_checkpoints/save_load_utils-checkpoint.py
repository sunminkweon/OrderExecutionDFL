# save_module.py

import pandas as pd
import os
import torch

def save_to_csv(pred, model_name, save_dir):
    """
    Save data to a CSV file.

    Parameters:
        pred, target (Tensor) : save prediction and target of predictions
        model_name (String) : model which makes the predictions
        save_dir (String) : save diretory
    Returns:
        None
    """
    predictions_file = os.path.join(save_dir, f'{model_name}_predictions.csv')
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
        
    pred_df = pd.DataFrame(pred)
    
    pred_df.to_csv(predictions_file, index=False)

def save_target_to_csv(save_dir, trg):
    trg_file = os.path.join(save_dir, 'targets.csv')
    
    if isinstance(trg, torch.Tensor):
        trg = pred.cpu().detach().numpy()
        
    trg_df = pd.DataFrame(trg)
    
    trg_df.to_csv(trg_file, index=False)
    
def load_target_from_csv(save_dir):
    trg_dir = os.path.join(save_dir, 'targets.csv')
    trg = pd.read_csv(trg_dir)
    return trg
    

def load_DA_predictions_from_csv(model_name, save_dir):
    """
    Load data from a CSV file.

    Parameters:
        model_name (str): Name of the CSV file.
    Returns:
        Prediction , target : pandas.DataFrame: Loaded data. 
    """
    pred_dir = os.path.join(save_dir, 'DA_' + f'{model_name}_predictions.csv')
    
    pred = pd.read_csv(pred_dir)
    return pred
    
def load_E2E_predictions_from_csv(model_name, save_dir):
    """
    Load data from a CSV file.

    Parameters:
        model_name (str): Name of the CSV file.
    Returns:
        Prediction , target : pandas.DataFrame: Loaded data. 
    """
    pred_dir = os.path.join(save_dir, 'E2E_' + f'{model_name}_predictions.csv')
    
    pred = pd.read_csv(pred_dir)
    return pred

def load_Naive_predictions_from_csv(model_name, save_dir):
    """
    Load data from a CSV file.

    Parameters:
        model_name (str): Name of the CSV file.
    Returns:
        Prediction , target : pandas.DataFrame: Loaded data. 
    """
    pred_dir = os.path.join(save_dir, 'Naive_' + f'{model_name}_predictions.csv')
    
    pred = pd.read_csv(pred_dir)
    return pred
