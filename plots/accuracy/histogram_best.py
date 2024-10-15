import sys
import os 
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.config import get_cfg_defaults
import argparse
import torch
from pathlib import Path
from models.utils import create_dir_name
import smplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

#some params
backbones = ('alexnet.yaml','vgg11.yaml','clip.yaml')
backbone_titles=('AlexNet','Vgg11','RN50x4')
percents=(1, 5, 10, 15, 20, 25, 50, 75, 100)
subjects=(1,2,5,7)
subject_colors=('b','g','r','c')
subject_ids=(1,2,5,7)
finetunes=(False,True)
corr_cut=-100


def get_corr(subjects, backbone, percent, finetune): #return corr of 
    corr=[]
    for sub in subjects:
        cfg=get_cfg_defaults()
        cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+backbone)
        opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
        cfg.merge_from_list(opts)
        corr_path=Path(str(create_dir_name(cfg,sub))+'/lightning_logs/version_%d/predictions/'%0)
        this_corr=torch.load(str(corr_path) + '/corr_combined.pt')
        corr.append(this_corr)
    return corr


models=[{'backbone':'alexnet.yaml', 'finetune':False, 'percent': 25,'color':'b', 'label': 'AlexNet-False-25'},
        {'backbone':'alexnet.yaml', 'finetune':True, 'percent': 10,'color':'g','label':'AlexNet-True-10'},
        {'backbone':'vgg11.yaml', 'finetune':False, 'percent': 5,'color':'r','label':'Vgg11-False-5'},
        {'backbone':'vgg11.yaml', 'finetune':True, 'percent': 1,'color':'c','label':'Vgg11-True-1'},
        {'backbone':'clip.yaml', 'finetune':False, 'percent': 1,'color':'m','label':'RN50x4-False-1'},
        {'backbone':'clip_text.yaml', 'finetune':False, 'percent': 100, 'color':'k','label':'Clip-text-100'}
        ]


fig,ax=plt.subplots(figsize=(5,5.7))
subjs=[1,2,5]


custom_labels=[]
custom_colors=[]

for model in models:
#model=models[0]
    corr=torch.cat(get_corr([1,2,5],model['backbone'],model['percent'],model['finetune']))
    plt.hist(corr,100,histtype='step',color=model['color'])
    custom_labels.append(model['label'])
    custom_colors.append(model['color'])

cfg=get_cfg_defaults()
custom_handles = [plt.Rectangle((0,0),1,1, color=color) for color in custom_colors]
plt.legend(custom_handles, custom_labels,loc='upper right',fontsize='xx-small')
plt.xlabel('Correlation')
plt.ylabel('Number of voxels')
out_path=Path(cfg.PATHS.NSD_PLOTS + '/accuracy/')
out_path.mkdir(parents=True,exist_ok=True)
plt.savefig(str(out_path) + '/accuracy-roi-text_histogram.png',dpi=800)
