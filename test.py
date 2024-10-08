import torch
from lightning_module import EndtoEndModelModule
from data_provider.EndtoEndDataModule import initialize_data_module
from utils import *
from trainer import Trainer
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from model import *
import torch

class Test_Trainer:
    def __init__(self, save_dir, model, args, step_size):
        self.save_dir =save_dir
        self.model = model
        self.args = args
        # self.end_to_end_model = torch.nn.Linear(args.prediction_length, args.prediction_length)
        self.accelerator = args.accelerator
        self.learning_rate = args.learning_rate
        self.step_size = step_size

        x = cp.Parameter(self.args.prediction_length)
        y = cp.Variable(self.args.prediction_length)
        left_vol = cp.Parameter()
        H = torch.eye(self.args.prediction_length)
        t = cp.Parameter(self.args.prediction_length)

        obj = cp.quad_form( cp.multiply( x,y), H ) 
        cons = [cp.sum(y) == left_vol, y >= 0, cp.multiply(y, t) == 0]
        prob = cp.Problem(cp.Minimize(obj), cons)
        
        self.e2e_layer = CvxpyLayer(prob, parameters=[x, left_vol, t], variables=[y])  # Corrected parameters

    def initialize_data_module(self, step) :
        train_lodaer, val_lodaer, test_loader, scaler = initialize_data_module(self.args, step)

        return train_lodaer, val_lodaer, test_loader, scaler
    
    def train_model(self):
        
        # Initialize trading module
        self.preds = None
        self.trgs = None
        for idx, step in enumerate(self.step_size) :
            self.train_loader, self.val_loader, self.test_loader, self.scaler = self.initialize_data_module(step)
            self.module = EndtoEndModelModule.LightningEndtoEndModel(self.model, self.e2e_layer , self.scaler, self.args.prediction_length, step, self.learning_rate, self.args.alpha, self.train_loader, self.val_loader, self.test_loader)
                                                            
            # Use the train_model function from the trainer module
            self.trainer = Trainer(self.module, self.args)

            predictor = self.trainer.test_model_tmp(self.test_loader)
            # Retrieve predictions after training
            self.prediction, self.target = self.module.return_prediction()

            if self.preds ==None :
                self.preds = self.prediction.contiguous().view(1,-1)
                self.trgs = self.target.contiguous().view(1,-1)
            else :
                self.preds = torch.cat([self.preds, self.prediction.contiguous().view(1,-1)], dim=0) 
                self.trgs = torch.cat([self.trgs, self.target.contiguous().view(1,-1)], dim=0) 
            
        return self.preds, self.trgs
        
    def save_prediction(self) :
        print(self.preds.shape)
        save_to_csv( self.preds.reshape(len(self.step_size),-1).detach().clone() , "E2E_" + self.model.__class__.__name__, self.save_dir)   
        
    def save_model_parameters(self) :
        # Save the model's state_dict to the specified file path
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "E2E_" + f'{self.model.__class__.__name__}_params.pth'))

    #def load_model_parameters() :

def test_models(save_dir,  model_list, args, configurations, step_size) :
    pred_results = []
    
    models = model_list
        
    for model in models :
        trainer = Test_Trainer(save_dir, model, args, step_size)
        pred, trg = trainer.train_model()
        #trainer.save_model_parameters()
        trainer.save_prediction()

        models.append(model)
        pred_results.append((pred,trg))
    return models, pred_results