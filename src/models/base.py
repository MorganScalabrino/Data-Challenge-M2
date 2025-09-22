"""
Module for building models.

This module provides a base abstract class `ModelBuilder` that defines the interface for model builders that appends head to TIMM backbone.
For either classification or object detection tasks, each builder should follow this interface to load saved models, set common attributes, and build the whole model.
"""

from abc import ABC, abstractmethod
import io

import timm
import torch


SUPPORTED_HEADS = []
        
class ModelBuilder(ABC):
    """Base abstract class for model builders.

    Defines the interface that each builder should follow to load saved models, set common attributes,
    and build the whole model. This class defines how to load a saved model and set common attributes.
    """

    def __init__(self):
        """Initializes the ModelBuilder instance with default values.

        Attributes:
            backbone (torch.nn.Module): The loaded backbone model.
            c (int): The number of channels in the input image.
            h (int): The height of the input image.
            w (int): The width of the input image.
            num_classes (int): The number of classes in the dataset.
        """
        self.backbone = None
        self.c = None
        self.h = None
        self.w = None
        self.num_classes = None

        return None

    # TODO: Remove it (never used apart in the test package)
    def load_from_memory(self, timm_model:torch.nn.Module): # Load a backbone
        self.backbone = timm_model
        return None

    def load_from_timm_database(self, modelname:str, num_classes=None, pretrained=True):
        """Download a Timm backbone model from the Hugging face database

        Args:
            modelname (str): _description_
            num_classes (_type_, optional): _description_. Defaults to None.
            pretrained (bool, optional): _description_. Defaults to True.
        """
        self.backbone = timm.create_model(modelname, num_classes=num_classes, exportable=True, pretrained=pretrained)
        return None


    def load_from_pth(self, modelpath:str, num_classes:int):
        """Load from a checkpoint

        Args:
            modelpath (str): Path to the checkpoint file.
            num_classes (int): Number of classes in the dataset.
        """
        with open(modelpath, "rb") as fhandler:
            buffer = io.BytesIO(fhandler.read())
            loaded_state_dict = torch.load(buffer, weights_only=True)
        model_name= "UNKNOWN"
        self.backbone = timm.create_model(model_name, num_classes=num_classes, exportable=True)	
        self.backbone.load_state_dict(loaded_state_dict, strict=True)
        return None


    def load_state_dict(self, state_dict, strict=False):
        """Update the state dict of a customized Timm model"""
        self.backbone.load_state_dict(state_dict, strict=strict)
        return None


    def set_head(self, head:str) -> None:
        """Set the detection head.

        Args:
            head (str): Name of the detection head.

        Raises:
            ValueError: Error if the head requested is not listed among supported heads
        """
        supported_heads = [item.lower() for item in SUPPORTED_HEADS]

        if head.lower() in supported_heads:
            self._head_type = head
        else:
            raise ValueError("Supported detection head are: {}".format(SUPPORTED_HEADS))
        return None


    def set_backbone(self, backbone:torch.nn.Module):
        """Add a backbone module to the inner classification module.

        Args:
            backbone (torch.nn.Module): a initialized backbone (timm expected).
        """        
        backbone.reset_classifier(0)
        self.backbone_model.add_module("backbone", backbone)        
        self.backbone_model.input_size = backbone.default_cfg["input_size"]
        self.backbone_model.architecture = backbone.default_cfg["architecture"] + "." + backbone.default_cfg["tag"]    
        self.backbone_model_set = True        
        return None


    @abstractmethod
    def build(self):
        """This method builds the whole model,ie timm backbone and a custom head.

        Note:
            This method must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def _get_features_dimensions(self):
        """Get the features dimensions of the model.

        Returns:
            tuple: The height and width of the feature map.

        Note:
            This method must be implemented by subclasses.
        """
        pass
