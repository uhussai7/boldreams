#Here we want the xmas plots (compares two models)
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.config import get_cfg_defaults
import argparse
import torch
from pathlib import Path
from models.utils import create_dir_name
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 22,
})

def xmas_plot(corr1,corr2,ax):
    y=np.asarray([corr1, corr2]).max(0)
    x=corr1-corr2
    ax.hexbin(x,y,cmap='inferno', bins=30,mincnt=1)
    #ax.set_xlim([-0.6,0.6])

def get_corr(backbone,subj,percent,finetune,version=0):
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'/'+backbone + '.yaml')
    opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
    cfg.merge_from_list(opts)
    cfg.freeze()
    corr_path=Path(str(create_dir_name(cfg,subj))+'/lightning_logs/version_%d/predictions/'%version)
    return torch.load(str(corr_path) + '/corr_combined.pt')

fig,ax=plt.subplots()
xmas_plot(get_corr('alexnet',1,1,False),get_corr('alexnet',1,100,False),ax)
plt.savefig('/home/u2hussai/scratch/test.png')
