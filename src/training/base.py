from abc import ABC, abstractmethod
import json
import math
import os
from typing import Dict, List, Tuple
import warnings

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, PolynomialLR, LambdaLR

from src.evaluation.base import IEvalHook
from src.boards import BaseBoard, TensorBoard
from src.__init__ import __version__
from .ema import EMA



class BaseTrainer(ABC):
    """Base class for fine-tuning a neural network model. 
    """

    def __init__(self,
                model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader = None,
                class_to_idx: dict = {},                
                savepath: str = ".",
                epochs: int = 10,
                learning_rate: float = 5e-4,
                optimizer: torch.optim.Optimizer=None,
                scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                batch_size:int=4,
                board: BaseBoard = None,
                last_epoch: int = -1,
                scoring_metric: str = "loss",
                test_freq: int = 1,
                full_training: bool=False):
        """
        Initializes the training process.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            train_dataloader (torch.utils.data.DataLoader): The data loader for the training set.
            test_dataloader (torch.utils.data.DataLoader, optional): The data loader for the testing set. Defaults to None.
            class_to_idx (dict, optional): A dictionary mapping class labels to indices. Defaults to {}.
            savepath (str, optional): The path where model checkpoints and logs will be saved. Defaults to ".".
            epochs (int, optional): The number of training epochs. Defaults to 10.
            learning_rate (float, optional): The initial learning rate for the optimizer. Defaults to 5e-4.
            optimizer (torch.optim.Optimizer, optional): The optimizer for the model parameters. Defaults to SGD.
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): A learning rate scheduler. Defaults to None.
            batch_size (int, optional): The batch size for training. Defaults to 4.
            board (BaseBoard, optional): A BaseBoard instance for logging metrics and visualizations. Defaults to None.
            last_epoch (int, optional): The index of the last epoch trained. Defaults to -1.
            scoring_metric (str, optional): The metric used to evaluate model performance. Defaults to "loss".
            test_freq (int, optional): The frequency at which to evaluate the model on the testing set. Defaults to 1.
            full_training (bool, optional): Whether to train the backbone of a model. Defaults to False.
        """        
        self.savepath = savepath
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.class_to_idx = class_to_idx
        self.scheduler = scheduler        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.test_freq=test_freq        
        self.frozen_parts = [] if full_training else ["backbone"]

        self._maximizing_metrics = []
        self._minimizing_metrics = ["test loss"]

        self.ema_decay = None
        self.perform_ema = False

        if board is None:            
            self.board = TensorBoard(os.path.join(self.savepath,"tensorboard"), verbosity=3)
        else:            
            self.board = board

        self.last_epoch = last_epoch
        if last_epoch==-1:
            self.start_epoch = 0
            
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        else :
            self.start_epoch = last_epoch

        self.epoch_loss = None
        self.test_loss = None
        self.evalhook = None
        self.iteration = 0
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.multi_gpu_flag = torch.cuda.device_count() > 1
        
        if scoring_metric.lower() not in (self._maximizing_metrics + self._minimizing_metrics):
            raise ValueError("Invalid '{}' metric. Metric must be in {}.".format(scoring_metric, (self._maximizing_metrics + self._minimizing_metrics)))
        else:
            self._best_metric = scoring_metric
            self._best_score = None
            self._best_epoch = 0

        return None

    def _update_ema_model_if_any(self):
        if self.perform_ema:
            self.ema_model.update_ema()

    @abstractmethod
    def _training_step(self):
        """override this method to implement your own training step, with input x and target
        """
        pass

    @abstractmethod
    def _test_hook(self, epoch:int):
        pass


    def train(self):
        """Full training operation, from model, scheduler and optimizer initializations to last epoch."""

        # if self.multi_gpu_flag:
        
        #     self.board.display("GPU count: {}; Enable multi-gpu data batching".format(torch.cuda.device_count()))
            
        #     self._adapt_model_to_multigpu()

        if self.perform_ema:
            self.ema_model = EMA(self.model, self.ema_decay)
            self.ema_model.to(self.device)

        self._freeze_layers(self.frozen_parts)        
        
        self.optimizer = self._set_optimized_layers(self.optimizer)

        self.model.to(self.device)
    
        self._set_scheduler()
        
        self.board.set_total_steps(self.epochs, initial=self.start_epoch)
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch_loss = 0

            self.board.add_scalar("Learning rate", self.scheduler.get_last_lr()[0], epoch+1)
            
            self.training_loop()
            
            self.board.add_scalar("Train loss", self.epoch_loss/self.iteration, epoch+1)

            if (epoch+1)%self.test_freq==0:
                
                self._freeze_layers(["backbone", "head"])
                
                metrics = self._test_hook(epoch+1)

                metrics["Test loss"] = self.test_loss/self.iteration

                self._update_board(metrics, epoch+1)

                self.save_test_results(os.path.join(self.savepath, "epoch_"+str(epoch+1)), self.evalhook)
                
                score = self._select_metric(self._best_metric, metrics)
                if self._record_best_metric(score, epoch+1):
                    self._best_epoch = epoch
                    self.save_test_results(os.path.join(self.savepath, "best_epoch"), self.evalhook)
                
                self._freeze_layers(self.frozen_parts)
                    
        return None


    def _record_best_metric(self, score:float, epoch:int) -> bool:
        """Store the best score and the corresponding epoch number.
        Save minimum value for minimizing metrics (like test loss) and maximum value for maximizing metrics (like precision)

        Args:
            score (float): value of the metric.
            epoch (int): epoch number.        

        Returns:
            bool: return True if it is the best epoch.
        """
        flag = False

        if self._best_score is None:
            self._best_score = score
            flag = True
        
        else:

            if self._best_metric.lower() in self._minimizing_metrics:
                self._best_score = min(self._best_score, score)
                flag = True

            if self._best_metric.lower() in self._maximizing_metrics:
                self._best_score = max(self._best_score, score)
                flag = True                
        
        return flag

    
    def _select_metric(self, name:str, metrics:Dict[str,float]) -> float:
        """Select the reference metrics for saving the best epoch.        

        Args:
            name (str): name of the selected metric.
            metrics (Dict[str,float]): A dictionary of metric values.

        Returns:
            float: value of the selected metric.
        """        
        metrics_recorded = dict()
        for key in metrics.keys():
            metrics_recorded[key.lower()] = key        

        if name.lower() in metrics_recorded.keys():
            metric = metrics[metrics_recorded[name]]
        else:
            raise ValueError("Invalid '{}' metric.".format(name))

        return metric
        

    def training_loop(self):
        """Training loop for a single epoch. This function updates the scheduler.
        """
        self.board.secondary_start(len(self.train_dataloader.sampler))

        self.iteration = 0
        
        for x, target in self.train_dataloader:
            self.iteration += x.size(0)
            self.board.secondary_progress.update(x.size(0))
            self._training_step(x.to(self.device), target.to(self.device))
            self._update_ema_model_if_any()
            

        if self.scheduler is not None:
            self.scheduler.step()

        self.board.secondary_progress.close()
        
        return None

    def _freeze_layers(self, targets: List[str]):
        """Freeze specified parts of the model.

        Args:
            targets (Lsit[int]): list of targets, "backbone" and/or "head"
        """        
        if "backbone" in targets:            
            self.model.backbone.train(False)
            self.model.backbone = self._set_requires_grad_layers(self.model.backbone, False)

        else:            
            self.model.backbone.train(True)
            self.model.backbone = self._set_requires_grad_layers(self.model.backbone, True)

        if "head" in targets:
            self.model.head.train(False)
            self.model.head = self._set_requires_grad_layers(self.model.head, False)
            
        else:
            self.model.head.train(True)
            self.model.head = self._set_requires_grad_layers(self.model.head, True)

        self.model.training = bool(self.model.backbone.training + self.model.head.training)
        
        return


    def _set_requires_grad_layers(self, module:torch.nn.Module, value:bool) -> torch.nn.Module:
        """Freeze all layers of a Module.

        Args:
            module (torch.nn.Module) : Module to freeze.
            value (bool) : True to freeze, False otherwise.

        Returns:
            torch.nn.Module : Module with all layers frozen.
        """        
        for parameter in module.parameters():
            if parameter.is_floating_point():
                parameter.requires_grad = value
        return module


    def _set_optimized_layers(self, optimizer:torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Set the parameters updated by the optimizer.

        Args:
            optimizer (torch.optim.Optimizer) : Optimizer to update.

        Returns:
            torch.optim.Optimizer : Optimizer with all layers updated.
        """
        optimized_parameters = list()
        
        for parameter in optimizer.param_groups[0]['params']:
            if parameter.requires_grad:
                optimized_parameters.append(parameter)

        optimizer.param_groups.clear()

        optimizer.state.clear()

        optimizer.add_param_group({'params' : optimized_parameters})

        return optimizer


    def _set_scheduler(self): 
        """Initialize a scheduler according to the specified options in the class constructor.
        """
        
        # set for a two order(x100) amplitude
        self.board.display("Scheduler: {}".format(self.scheduler))
        if self.scheduler.lower() == "triangular2".lower():
            base_lr = self.lr /10
            max_lr = self.lr
            step_size_up = self.epochs / 10 # might be a variable
            self.scheduler = CyclicLR(self.optimizer, base_lr, max_lr, step_size_up=step_size_up, mode="triangular2")

        elif self.scheduler.lower() == "triangular2inv".lower():
            base_lr = self.lr /10
            max_lr = self.lr
            step_size_up = self.epochs / 10 # might be a variable
            self.scheduler = CyclicLR(self.optimizer, max_lr, base_lr, step_size_up=step_size_up, mode="triangular2")


        elif self.scheduler.lower() == "ExponentialLR".lower():
            gamma = (-2/self.epochs)*math.log(10)# might be a variable
            self.scheduler = ExponentialLR(self.optimizer, gamma)

        elif self.scheduler.lower() == "PolynomialLR".lower():
            self.scheduler = PolynomialLR(self.optimizer, total_iters=self.epochs)

        elif self.scheduler.lower() == "constant.lower()":
            lambda_constant = lambda epoch: 1.0
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_constant)

        self.board.display("Base Learing rate: {}".format(self.lr))

        if self.start_epoch > 0:
            self.scheduler.last_epoch = self.last_epoch-1

            self.scheduler.step()        

        else:
            self.scheduler.last_epoch = self.last_epoch

        return None


    def _create_output_folder(self, destination:str):
        """Creates a directory at the specified destination.
        Emits warnings if directory already exists.

        Args:
            destination (str): destination path.
        """
        if os.path.isdir(destination):
            warnings.warn("Directory {} already exists".format(destination))
        os.makedirs(destination, exist_ok=True)
        
        return None

    def _update_board(self, metrics:Dict[str,float], epoch:int):
        """Update the Tensorboard with metrics.

        Args:
            metrics (Dict[str,float]): A dictionary of metric values.
            epoch (int): current epoch.
        """        
        for metric_name, value in metrics.items():
            
            self.board.add_scalar(metric_name, value, epoch)

        return None

    @abstractmethod
    def _save_configuration_file(self, destination:str, checkpoint_name:str):
        """Save the configuration file of the model.
        """
        

    def save_test_results(self, destination:str, evalhook:IEvalHook):
        """Save the model and the results of the evaluation hook.

        Args:
            destination (str): folder where model and evaluation results are saved. 
            evalhook (IEvalHook): evaluation hook that records test results and computes metrics.
        """
        self._create_output_folder(destination)

        evalhook.write(destination)

        if self.perform_ema:
            self.save_as_pth(self.ema_model.shadow, destination)
        else:
            self.save_as_pth(self.model, destination)

        self.save_optimizer_state(self.optimizer, destination)

        self.save_as_onnx(self.model, destination)# warning: this freeze the whole model
        return None
    

    def save_as_pth(self, model:torch.nn.Module, destination:str):
        """Save a model as a pth file (pytorch format) and its configuration file.        

        >>> config = {
        >>>     "model": "convnextv2_atto.fcmae_ft_in1k",
        >>>     "head": "DropConnectLinearHead",
        >>>     "checkpoint": "model.pth",
        >>>     "classes names": [
        >>>         "acari",
        >>>         "annelida",
        >>>         "myriapoda",
        >>>     ],
        >>>     "version": "0.3.4",
        >>>     "optimizer": "AdamW_optimizer.pth"
        >>> }

        Args:
            model (torch.nn.Module): any model.
            destination (str): path to the save folder.
        """
        checkpoint_name = "model.pth"

        self._save_configuration_file(destination, checkpoint_name)

        torch.save(model.state_dict(), os.path.join(destination, checkpoint_name))	
        return None


    def save_as_onnx(self, model:torch.nn.Module,  destination:str):
        """Save a model as a ONNX file (cross-library format).

        Args:
            model (torch.nn.Module): any model.
            destination (str): path to the save folder.
        """
        path = os.path.join( destination, "model.onnx")
        dummy_input = torch.randn(4, *self.model.input_size).to(self.device)
        input_names = [ "input" ]
        output_names = [ "output" ]
        if self.multi_gpu_flag:
            torch.onnx.export(self.model.module, dummy_input, path, verbose=False, input_names=input_names, output_names=output_names)            
        else:
            torch.onnx.export(self.model, dummy_input, path, verbose=False, input_names=input_names, output_names=output_names)
        return None


    def save_optimizer_state(self, optimizer:torch.optim.Optimizer, destination:str):
        """Save the optimizer inner state. Use it for pursuing the training of a model.

        Args:
            optimizer (torch.optim.Optimizer):any optimizer.
            destination (str): path to the save folder.
        """
        with open(os.path.join(destination, "config.json")) as fhandler:
            config_dict = json.load(fhandler)
        
        optim_entry = dict()
        optim_entry["optimizer"] = optimizer.__class__.__name__ + "_optimizer.pth"
        config_dict.update(optim_entry)
        with open(os.path.join(destination, "config.json"), "w") as fhandler:
            json.dump(config_dict, fhandler, sort_keys=False, indent=4)
        torch.save(optimizer.state_dict(), os.path.join(destination, config_dict["optimizer"]))
        return None

    def _adapt_model_to_multigpu(self):
        """Change model attributes to fit with multiGpu constraints.
        """
        self.model = torch.nn.DataParallel(self.model)
        
        self.model.backbone = self.model.module.backbone

        self.model.head = self.model.module.head

        self.model.loss_function = self.model.module.loss_function

        self.model.architecture =  self.model.module.architecture

        self.model.input_size = self.model.module.input_size
        
        return None
