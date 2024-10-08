import optuna
import pytorch_lightning as pl
from lightning_module import ModelModule
import numpy as np
from trainer import Trainer
from model import *
import os
import yaml

class HPO_trainer:
    def __init__(self, model_class, configurations, train_loader, val_loader, max_epochs, accelerator):
        self.model_class = model_class
        self.configurations = configurations
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        
    def objective(self, trial):
        # Define hyperparameters to optimize
        hyperparameters = {
            "learning_rate": "learning_rate",
            "num_layers": "num_layers",
            "hidden_dim": "hidden_dim",
            "drop_out": "drop_out",
            "patch_len": "patch_len",
            "stride": "stride",
            "kernel_size": "kernel_size",
            "n_heads": "n_heads"
        }
        
        # Check if each attribute exists in the configurations object and update if found
        for param_name, attr_name in hyperparameters.items():
            if hasattr(self.configurations, attr_name):
                if param_name == "learning_rate":
                    setattr(self.configurations, attr_name, trial.suggest_float(param_name, 1e-5, 1e-2, log=True))
                elif param_name == "num_layers":
                    setattr(self.configurations, attr_name, trial.suggest_int(param_name, 1, 3))
                elif param_name == "hidden_dim":
                    hidden_dim_choices = [2**i for i in range(4, 10)]
                    setattr(self.configurations, attr_name, trial.suggest_categorical(param_name, hidden_dim_choices))
                elif param_name == "drop_out":
                    drop_out_choices = [0.1, 0.2, 0.3, 0.4, 0.5]
                    setattr(self.configurations, attr_name, trial.suggest_categorical(param_name, drop_out_choices))
                elif param_name in ["patch_len", "stride"]:
                    patch_len_choices = [2**i for i in range(2, 4)]
                    setattr(self.configurations, attr_name, trial.suggest_categorical(param_name, patch_len_choices))
                elif param_name == "kernel_size":
                    kernel_size_choices = [i for i in range(1, 79, 2)]
                    setattr(self.configurations, attr_name, trial.suggest_categorical(param_name, kernel_size_choices))
                elif param_name == "n_heads":
                    n_heads_choices = [2**i for i in range(2, 4)]
                    setattr(self.configurations, attr_name, trial.suggest_categorical(param_name, n_heads_choices))
                    
        # Instantiate and train the model
        config = self.configurations
        config.device = self.accelerator
        
        model_instance = self.model_class(config)
        
        module_type= ModelModule.LigtningModel
            
        module = module_type(model_instance, self.configurations.learning_rate)
        # Use the train_model function from the trainer module
        trainer = Trainer(module)

        # Call train_model method
        val_loss = trainer.train_model(
            accelerator=self.accelerator,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            max_epochs=self.max_epochs,
            dirpath=None  # Change this to the desired directory path if needed
        )
        
        # Return validation loss for optimization
        return val_loss

    def optimize(self, n_trials=1, timeout=10800):
        # Define Optuna study
        study = optuna.create_study(direction="minimize")
        
        # Run the optimization
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        # Get the best hyperparameters
        best_params = study.best_params
            
        
        return best_params
        
def optimize_hyperparameters(save_dir, configurations, train_loader, val_loader, accelerator, max_epochs):
        # Define hyperparameter optimizer
    for model_name, config in configurations.items():
        # Define hyperparameter optimizer
        model_class = eval(model_name)
        optimizer = HPO_trainer(
            model_class=model_class,
            configurations=configurations[model_name],
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            accelerator=accelerator
        )
        # Perform hyperparameter optimization
        best_params = optimizer.optimize()
        
        for key, value in best_params.items():
            configurations[model_name].set(key, value)
            
    all_hyperparams_file_path = os.path.join(save_dir, 'best_hyperparameters.yml')
    with open(all_hyperparams_file_path, 'w') as file:
        yaml.dump(configurations, file)

    return configurations
