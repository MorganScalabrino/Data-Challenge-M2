import os
import json
from typing import Dict, List
import warnings



from src.evaluation import ClassificationEvalHook, HierarchicalClassificationEvalHook
from src.boards import BaseBoard
from src.hxe_loss import HierarchyLoss
from .base import BaseTrainer
from src.__init__ import __version__

class FineTuningTrainer(BaseTrainer):
    """Fine-tuning trainer class for a classification model. This class is used to train a model with a custom classification head.
    """

    def __init__(self,
        savepath: str,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        class_to_idx:dict,
        scheduler: str="constant",
        optimizer: torch.optim.Optimizer=SGD,
        loss:torch.nn.Module=None,
        epochs:int=100,
        batch_size:int=4,
        learning_rate:float=1e-4,
        test_freq:int=10,
        board:BaseBoard=None,
        last_epoch:int=-1,
        scoring_metric="test loss",
        full_training: bool=False
        ):
        """
        Initialize the trainin g process.

        Args:
            savepath (str): _description_
            model (torch.nn.Module): _description_
            train_dataloader (torch.utils.data.DataLoader): _description_
            test_dataloader (torch.utils.data.DataLoader): _description_
            class_to_idx (dict): _description_
            scheduler (str, optional): _description_. Defaults to "constant".
            optimizer (torch.optim.Optimizer, optional): _description_. Defaults to SGD.
            loss (torch.nn.Module, optional): _description_. Defaults to None.
            epochs (int, optional): _description_. Defaults to 100.
            batch_size (int, optional): _description_. Defaults to 4.
            learning_rate (float, optional): _description_. Defaults to 1e-4.
            test_freq (int, optional): _description_. Defaults to 10.
            board (BaseBoard, optional): _description_. Defaults to None.
            last_epoch (int, optional): _description_. Defaults to -1.
            scoring_metric (str, optional): _description_. Defaults to "test loss".
            full_training (bool, optional): _description_. Defaults to False.           
        """

        super(FineTuningTrainer, self).__init__(
            savepath=savepath,
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            class_to_idx=class_to_idx,
            scheduler=scheduler,
            optimizer=optimizer,            
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            test_freq=test_freq,
            board=board,
            last_epoch=last_epoch,
            scoring_metric=scoring_metric,
            full_training=full_training,
            )

        self._maximizing_metrics = ["ap", "roc auc", "precision", "recall"]
        self._minimizing_metrics = ["test loss"] 
        self.scoring_metric = scoring_metric
        self.class_to_idx = class_to_idx
        
        self.loss = CrossEntropyLoss()

        self.evalhook = ClassificationEvalHook(self.class_to_idx, scenario="multiclass")

        if loss is not None:
            warnings.warn("Loss function: Overriding {} with user specified {}".format(self.loss, loss))
            self.loss = loss()

        self.softmax = Softmax()

        return None


    def _training_step(self, x:torch.Tensor, target:torch.Tensor):
        """Training step for a single iteration.

        Args:
            x (torch.Tensor): input tensor.
            target (torch.Tensor): groundtruth tensor.
        """
        self.optimizer.zero_grad()
        
        y = self.model(x)        
        
        loss = self.loss(y, target)
        
        loss.backward()

        self.optimizer.step()

        self.optimizer.zero_grad()

        self.epoch_loss += loss.item()*x.size(0)
        
        return None


    def _get_val_model(self):
        if self.perform_ema:
            return self.ema_model

        return self.model

    def _test_hook(self, epoch) -> Dict[str, float]:
        """Test hook over the whole test dataset.

        Returns:
            Dict[str, float]: metrics dictionary.
        """
        
        self.iteration = 0

        self.model.eval()
        if self.perform_ema:
            self.ema_model.eval()

        val_model = self._get_val_model()

        self.evalhook.reset()

        self.test_loss = 0

        with torch.no_grad():            
            for x, target in self.test_dataloader:
                
                self.iteration += x.size(0)

                x = x.to(self.device)

                target = target.to(self.device)

                y = val_model(x)

                loss = self.loss(y, target)

                self.test_loss += loss.item()*x.size(0)

                y = self.softmax(y)

                self.evalhook.append(y.cpu(), target.cpu())

        self.evalhook.confusion_matrix()

        metrics = dict()

        metrics["AP"] = self.evalhook.average_precision()

        metrics["ROC AUC"] = self.evalhook.auroc()

        metrics["Precision"] = self.evalhook.precision()
        
        metrics["Recall"] = self.evalhook.recall()        
        
        return metrics


    def _save_configuration_file(self, destination:str, checkpoint_name:str):
        """Save a JSON storing hyperparameters 

        >>> config = {
        >>>     "model": "convnextv2_atto.fcmae_ft_in1k",
        >>>     "head": "DropConnectLinearHead",
        >>>     "checkpoint": "model.pth",
        >>>     "classes names": [
        >>>         "acari",
        >>>         "annelida",
        >>>         "myriapoda",
        >>>     ],
        >>>     "version": "0.5.1",
        >>>     "optimizer": "AdamW_optimizer.pth"
        >>> }
        Args:
            checkpoint_name (str): Name of the checkpoint file (e.g. checkpoint.pth).
        """
        config_dict = dict()
        
        config_dict["model"] = self.model.architecture

        config_dict["head"] = self.model.head_type

        config_dict["checkpoint"] = "model.pth"

        config_dict["resize"] = self.train_dataloader.base_resize

        config_dict["classes names"] = list(self.class_to_idx.keys())

        config_dict["version"] = __version__

        with open(os.path.join(destination, "config.json"), "w") as fhandler:
            json.dump(config_dict, fhandler, sort_keys=False, indent=4)
        
        return None
