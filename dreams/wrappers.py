import torch
from torch.nn.modules.module import Module
from torch.nn import ModuleDict
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch.nn import ReLU
from torch.nn.functional import softmax,silu,sigmoid,leaky_relu,relu
from torch.nn import Threshold
import numpy as np
from torchvision.transforms import GaussianBlur,Compose,RandomAffine

class roi_extractor(Module):
    """
    Module to extract one roi
    """
    def __init__(self,roi):
        """
        Initializer
        :param roi: The roi filter of size Nv
        """
        super(roi_extractor, self).__init__()
        self.roi=roi #these are of size visual cortex

    def forward(self,fmri):
        """
        Forward
        :param fmri: Bold signal of size [1,Nv]
        :return: Filtered result
        """
        return fmri[:,self.roi]
    
class dream_wrapper(Module):
    """
    Model wrapper
    """
    def __init__(self,model,roi_dic):
        """
        Initializer
        :param model: Model the predicts the fmri signal
        :param roi_dic: dictonary containing roi names and rois
        """
        super(dream_wrapper, self).__init__()
        self.model=model
        self.roi_dic=roi_dic
        self.roi=ModuleDict({key:roi_extractor(roi=roi_dic[key]) for key in roi_dic})

    def forward(self,x):
        """
        Forward
        :param x: Image batch of shape [B,3,h,w]
        :return: Signal in each roi
        """
        fmri=self.model(x)
        return {e:self.roi[e](fmri) for e in self.roi}#,self.img_e(x)