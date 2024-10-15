#just perform retinotopic dreams the same way did abstract dreams
#this should really be part of abstract dreams but those jobs are in queue

import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#from dreams import wrappers
from models import modules
from configs.config import get_cfg_defaults
from nsdhandling.core import NsdData
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from models.utils import create_dir_name
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from dreams.wrappers import dream_wrapper 
from dreams.objectives import roi as roi_
from dreams.objectives import diversity
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from lucent.optvis import render, param, objectives
from torch.optim import Adam,SGD
from tqdm import tqdm
from pathlib import Path
import nibabel as nib

def load_model(cfg,subj): #load the model
    #get the base config
    #handle devices
    accelerator='cpu'
    N_devices=1
    strategy='ddp_find_unused_parameters_true'
    if torch.cuda.is_available():
        accelerator='gpu'
        N_devices=int(torch.cuda.device_count())
    #load the subject data
    nsd_data=NsdData([subj])
    nsd_data.load_preprocessed(cfg.BACKBONE.INPUT_SIZE)    #handle text or not for data_loaders
    #handle text or not for data_loaders
    if cfg.BACKBONE.TEXT == True:
        import clip
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE,text=True,tokenizer=clip.tokenize)
    else:
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE)
     #load the checkpoint
    Nv=int(nsd_data.data[0]['Nv']) #number of voxels
    checkpoint_path=str(create_dir_name(cfg,subj))+'/lightning_logs/version_%d/checkpoints/'%0
    checkpoints=os.listdir(checkpoint_path)
    print('These are the checkpoints',checkpoints)
    epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
    max_epoch_ind=np.argsort(epochs)[-1]
    max_epoch=epochs[max_epoch_ind]
    resume_checkpoint=checkpoint_path+checkpoints[max_epoch_ind]
    print('Loading checkpoint:',resume_checkpoint)
    enc=modules.LitEncoder.load_from_checkpoint(resume_checkpoint,cfg=cfg,data_loader=nsd_data.data_loaders_train[0])
    return enc,cfg,nsd_data 

def choose_objective(roi,objective='default',img=None):
    roi='roi_'+roi
    if objective=='default':
        return roi_(roi)
    if objective=='diversity':
        return roi_(roi) - 2e-5* diversity(roi)

def dreaming_setup(enc,cfg,nsd_data,roi,rois,objective,Ndreams=4): #setup the dreaming #note this is different from abstract
    #rois=nsd_data.data[0]['roi_dic_combined'].item()
    dreamer=dream_wrapper(enc,rois)
    #optimizer=lambda params: SGD(params,lr=cfg.DREAMS.LR)
    optimizer=lambda params: Adam(params,lr=cfg.DREAMS.LR)
    param_f = lambda: param.image(cfg.BACKBONE.INPUT_SIZE[0], fft=True, decorrelate=True,sd=0.01,batch=Ndreams)
    jitter_only= [RandomAffine(cfg.DREAMS.ROTATE,translate=cfg.DREAMS.TRANSLATE,scale=cfg.DREAMS.SCALE, fill=0.0)]   
    obj = objective#roi_(roi) - 2e-5* diversity(roi) #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([-2]).cuda())
    _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
                    optimizer=optimizer,fixed_image_size=cfg.BACKBONE.INPUT_SIZE[0],thresholds=cfg.DREAMS.THRESHOLDS,show_image=False)
    return _


def mean_fmri_per_roi_per_image(enc, nsd_data,batch_size=4): #excludes prf
    #returns top images in an roi (save the index, check if index exists)
    roi_dic=nsd_data.data[0]['roi_dic_combined'].item()
    imgs=nsd_data.data_loaders_val_single[0].dataset.tensors[0]
    roi_img_means={key:np.zeros(len(imgs)) for key in list(roi_dic.keys())}
    for i in tqdm(range(0,len(imgs),batch_size)):
        img=imgs[i:i+batch_size]
        this_fmri=enc(img.cuda())
        for roi in roi_dic.keys():
            roi_img_means[roi][i:i+batch_size]=this_fmri[:,roi_dic[roi]].mean(-1).detach().cpu().numpy()
    return roi_img_means