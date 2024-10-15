#imports
from torch.nn import Module, Parameter, ParameterList, Linear, init
from torch.nn.functional import mse_loss
from torch import matmul,optim
import torch
import math
import os
from torchvision.models.feature_extraction import create_feature_extractor
from .utils import map_times_rf,layer_shapes,unique_2d_layer_shapes, channels_to_use,channel_summer,create_dir_name
import lightning as L
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.optim import Adam, SGD
from pathlib import Path
from yacs.config import CfgNode
import torchextractor as tx

class EncoderTrainer(L.Trainer):
    """
    Class for training the encoder
    """
    def __init__(self,cfg,subj,*args,**kwargs): #we do per subject
        self.cfg=cfg
        self.subj=subj
        self.out_path=create_dir_name(cfg,subj)
        super().__init__(*args,
                        default_root_dir=self.out_path,**kwargs)
        self.write_cfg_yaml()

    def write_cfg_yaml(self):
        with open(str(self.out_path)+'/config.yaml', 'w') as f:
            f.write(self.cfg.dump())

class LitEncoder(L.LightningModule):
    """
    Lightning wrapper for Encoder class
    """
    def __init__(self,cfg,data_loader,*args,**kwargs):        
        super().__init__()  
        self.encoder=Encoder(cfg,data_loader,*args,**kwargs)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.encoder(x)
        loss = mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        #print(loss)
        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return torch.stack((self.encoder(x),y))

    def configure_optimizers(self):
        if self.encoder.cfg.BACKBONE.FINETUNE == True: #train backbone or not
            optimizer = Adam(params=self.parameters(), lr=0.0001)
        else:
            if self.encoder.cfg.BACKBONE.TEXT == True:
                params_to_pass=self.encoder.readout.w.parameters()
            else: #
                params_to_pass=[
                                {'params': self.encoder.readout.rfs.parameters()},
                                {'params': self.encoder.readout.w},
                                {'params': self.encoder.readout.b}
                                ]
            optimizer = Adam(params=params_to_pass, lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[2, 3, 4, 5, 6, 7], gamma=0.8)
        return [optimizer], [scheduler]

    def forward(self,x):
        return self.encoder(x)

class EncoderReadOut(Module):
    """
    Module for fmri encoder readout
    """
    def __init__(self,cfg,feature_extractor,
                 rfs, #instance of ReceptiveField
                 N_channels):
        #TODO: need descriptor for this __init__ function
        super().__init__()
        self.cfg=cfg
        self.feature_extractor=feature_extractor
        self.rfs=rfs
        self.N_channels=N_channels
        self.Nv = self.rfs.Nv

        self.w=Parameter(torch.empty([self.N_channels,self.Nv]))
        self.b=Parameter(torch.empty(self.Nv))

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.b)
        init.kaiming_uniform_(self.w,a=math.sqrt(5))

    def forward(self,x):
        features=self.feature_extractor(x)
        sigma=self.rfs(features) #shape [B,N_channels,Nv]
        return (sigma[:,:,:]*self.w[None,:,:]).sum(-2) + self.b #summing the channel dimension here

class EncoderTextReadOut(Module):
    """
    Module for fmri text encoder readout 
    """
    def __init__(self,cfg,feature_extractor,Nv):
        """
        Initialization function for text encoder readout

        Args:
            cfg: config node
            feature_extractor: clip transformer branch
            Nv: number of voxels
            Nt: number of output neurons for the last layer of the clip transformer branch
        """
        super().__init__()
        self.cfg=cfg
        self.Nv=Nv
        self.Nt=int(cfg.BACKBONE.NT)
        self.feature_extractor=feature_extractor
        self.w=Linear(self.Nt,self.Nv)

    def forward(self,x):
        """
        Forward function for text fmri encoder

        Args:
            x: TODO: is x the tokenized version of the caption [B,N_captions_per_img,Lt]
        """
        #TODO: what to do with the multiple captions per image? Average them
        N_batch,N_captions_per_img,N_token=x.shape
        x=self.feature_extractor(
                                  x.view(N_batch*N_captions_per_img,-1)
                                ).view(N_batch,N_captions_per_img,self.Nt).mean(dim=1)
        return self.w(x.float())

class Encoder(Module): 
    """
    Module for fmri encoder
    """
    def __init__(self,cfg,data_loader=None,imgs=None,Nv=None):
        #TODO: need descriptor for this __init__ function and some comments below
        super().__init__()
        self.cfg=cfg #params
        self.imgs=imgs
        self.data_loader=data_loader
        self.Nv=list(data_loader)[0][1].shape[-1] if Nv is None else Nv
        self.Nt=cfg.BACKBONE.NT
        
        self.text_or_vision() #init function calls

    def text_or_vision(self):
        """
        Handle text or vision if using clip
        """
        if self.cfg.BACKBONE.TEXT is True:
            self.text_features=ImageFeatures(self.cfg)
            self.readout=EncoderTextReadOut(self.cfg,self.text_features,self.Nv)
        else:
            self.image_features=ImageFeatures(self.cfg)
            self.get_rf_sizes()
            self.get_channel_basis(self.get_imgs()) #TODO: this is done without cuda at initialization
            self.rfs=ReceptiveField(self.Nv,self.rf_sizes,self.layer_to_rf_size,self.channel_basis)
            self.readout=EncoderReadOut(self.cfg,self.image_features,self.rfs,self.N_channels)

    def get_imgs(self,N_imgs=15000):
        """
        Extracts images from dataloader if self.imgs is None, will extract up to N_imgs
        """
        print('Extracting images')
        if self.imgs is not None:
            return self.imgs
        else:
            N_imgs=min(N_imgs,self.data_loader.dataset.tensors[0].shape[0])
            return self.data_loader.dataset.tensors[0][:N_imgs]
        
    def get_rf_sizes(self):
        """Gets the unique rf-sizes to make receptive fields"""
        print('Calculating rf sizes')
        self.layer_shps=layer_shapes(self.image_features,(1,3)+self.cfg.BACKBONE.INPUT_SIZE)
        (
            self.rf_sizes, #unique rf_sizes
            self.layer_to_rf_size, #index of rf_size for each layer 
            self.channels #channels in each layer

        ) = unique_2d_layer_shapes(self.cfg.BACKBONE.LAYERS_TO_EXTRACT,self.layer_shps)
        print('Done')

    def get_channel_basis(self,imgs):
        print('Computing channel basis')
        self.channel_basis=channels_to_use(layers_2d=self.cfg.BACKBONE.LAYERS_TO_EXTRACT,
                                           feature_extractor=self.image_features,
                                           imgs=imgs,
                                           max_percent=self.cfg.BACKBONE.PERCENT_OF_CHANNELS)
        self.N_channels=channel_summer(self.channel_basis)
        print('Done')

    def forward(self,x):
        return self.readout(x)

class ImageFeatures(Module):
    """
    Module to extract features from a backbone
    """
    def __init__(self,cfg):
        """
        Intialization function for ImageFeatures class

        Args:
            cfg: yacs configuration node
        """
        super().__init__()
        self.cfg=cfg
        self.get_backbone()
        self.get_feature_extractor()

    def get_backbone(self):
        """Loads backbone architecture and weights (from one file)"""
        self.backbone_file=os.path.join(self.cfg.PATHS.BACKBONE_FILES + self.cfg.BACKBONE.FILE)
        try:
            print('Loading backbone from:', self.backbone_file)
            self.backbone=torch.load(self.backbone_file, map_location='cpu')
        except:
            raise Exception('Failed to load backbone')
        
        #figure out requires grad
        if self.cfg.BACKBONE.FINETUNE==False:
            print('Backbone fine tune is off, turning off grad for backbone')
            for param in self.backbone.parameters():
                param.requires_grad=False 

    def get_feature_extractor(self):
        """Creates feature extractor using torchvision's feature_extraction module"""
        if self.cfg.BACKBONE.TEXT:
            self.feature_extractor=self.backbone.encode_text
        elif 'clip' in self.cfg.BACKBONE.NAME:
            self.feature_extractor=create_feature_extractor(self.backbone.visual,return_nodes=self.cfg.BACKBONE.LAYERS_TO_EXTRACT)
        else:
            self.feature_extractor=create_feature_extractor(self.backbone,return_nodes=self.cfg.BACKBONE.LAYERS_TO_EXTRACT)

    def forward(self,x):
        return self.feature_extractor(x)

class ReceptiveField(Module):
    """
    Module for a receptive field, takes activative maps and multiplies with rf-fields for each voxel
    """
    def __init__(self,Nv,rf_sizes,layer_to_rf_size,channel_basis):
        """
        Initialization function
        TODO: fix this comment section
        Args:
            Nv: number of voxels
            rf_sizes: list of rf sizes [(H_1,W_1), (H_2,W_2), (H_3,W_3), ...]
        """
        super().__init__()
        self.Nv=Nv
        self.rf_sizes=rf_sizes
        self.rfs=ParameterList([Parameter(torch.empty((Nv,)+hw)) for hw in self.rf_sizes])
        self.layer_to_rf_size=layer_to_rf_size
        self.channel_basis=channel_basis

        self.reset_parameters()

    def reset_parameters(self):
        for rf in self.rfs:
            init.kaiming_uniform_(rf,a=math.sqrt(5))

    def forward(self,x):
        """
        Forward function for ReceptiveField class

        Args:
            x: output of ImageFeatures class
        Returns:
            tensor: Tensor with size [B,Nv,C_0+C_1+...] = [B,Nv,C_total]
        """
        out=[]
        for key,x_ in x.items():
            x_=x_[:,self.channel_basis[key],:,:]
            rf=self.rfs[self.layer_to_rf_size[key]]
            out.append(map_times_rf(x_,rf))
        return torch.cat(out,dim=1)

class TextFeatures(ImageFeatures): #seems like taking only the last layer is sufficient so this class is not useful for now
    """
    Module to extract clip text features
    """
    def __init__(self,cfg):
        """
        Initialization function

        Args:
            cfg: yacs configuration node
        """
        super().__init__(cfg)
    
    def get_feature_extractor(self): #only thing thats changed is that we are using tx library instead of torchvision
        self.feature_extractor=tx.Extractor(self.backbone,cfg.BACKBONE.LAYERS_TO_EXTRACT)

class integrated_gradient(Module):
    """
    A simple module to compute integrated gradients
    """
    def __init__(self,feature_ext):
        """
        Initializer
        :param model: The wrapped fmri encoding model
        """
        super(integrated_gradient, self).__init__()
        self.feature_ext=feature_ext

    def forward(self,img_in,roi,I_0=None,steps=10):

        img=torch.clone(img_in.detach())
        img.requires_grad=True
        optimizer=Adam(lr=2e-2,params=[img])

        if I_0 is None:
            I_0=torch.zeros_like(img)
            I_0.requires_grad=True

        ig = []
        alphas = torch.linspace(0, 1, steps)
        for alpha in alphas:
            imgp=I_0 + alpha*(img-I_0)
            features=self.feature_ext(imgp)
            features[:,roi].mean().backward()
            ig.append(img.grad)

        return torch.stack(ig).detach().cpu().abs().mean([0,1])


