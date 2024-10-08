import torch
from lightning_module import ModelModule
from trainer import Trainer
from utils import save_to_csv
from model import *
import os

class Prediction_Trainer:
    def __init__(self, save_dir, model_name, config, train_loader, val_loader, scaler, 
                 accelerator,  max_epochs, ticker):
        self.save_dir = save_dir
        self.model_name =  model_name
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler
        self.max_epochs = max_epochs
        
        self.ticker = ticker
        self.accelerator = accelerator
    
    def train_model(self):
    # Instantiate the model
        model_class = eval(self.model_name)
        self.model = model_class(self.config)
       # Determine the module type based on the model class
        module_type = ModelModule.LigtningModel
        
        # Initialize the module with the model and learning rate from the config
        self.module = module_type(self.model, self.config.learning_rate)
        
        # Set the directory path for saving model parameters
        dirpath = f"{self.save_dir}/{self.model.__class__.__name__}"
        
        # Initialize the trainer
        self.trainer = Trainer(self.module, self.config, self.ticker)
        
        # Call the train_model method
        val_loss = self.trainer.train_model(
            accelerator=self.accelerator,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            max_epochs=self.max_epochs,
            dirpath=dirpath  # Change this to the desired directory path if needed
        )
    
        return self.model

    def test_model(self, test_loader) :
        predictor = self.trainer.test_model(test_loader)
        # Retrieve predictions after training
        self.prediction, self.target = self.module.return_prediction()
        
        return self.prediction, self.target

    def save_model_parameters(self):
        # Save the model's state_dict to the specified file path
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{self.model_name}_params.pth'))

    def save_prediction(self) :
        save_to_csv( self.scaler.inverse_transform(self.prediction.squeeze().reshape(-1, 1)), "Naive_" + self.model.__class__.__name__, self.save_dir)
    
def training_prediction_models(save_dir, configurations, train_loader, val_loader, test_loader, scaler, accelerator, max_epochs, ticker) :
    pred_results = []
    models = []
    for model_name, config in configurations.items():
        trainer = Prediction_Trainer(save_dir, model_name, config, train_loader, val_loader, scaler, accelerator, max_epochs, ticker)
        model = trainer.train_model()
        pred, trg = trainer.test_model(test_loader)
        trainer.save_model_parameters()
        trainer.save_prediction()

        models.append(model)
        pred_results.append((pred,trg))
    return models, pred_results
    
    