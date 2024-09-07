import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.config import get_cfg_defaults
import argparse
import torch
from pathlib import Path
from models.utils import create_dir_name


#lets make the alexnet accuract plot
parser = argparse.ArgumentParser(
        description="Script for training encoders"
    )

parser.add_argument(
    '-c','--config',
    type=str,
    help='Config file name, if not done, please define config folder as environment variable named NSD_CONFIG_PATH'
)


parser.add_argument(
    '-v', '--version',
    type=int,
    default=0,
    help='If continuing from last checkpoint provide the version'
)

args=parser.parse_args()
cfg=get_cfg_defaults()
cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+args.config)


#the trend we are after is percentage
percents=(1, 5, 10, 15, 20, 25, 50, 75, 100)
subjects=(1,2,5,7)
finetunes=(True,False)

corr_cut=0.4

#load
corrs_mean = []
corrs_std = []
subj=subjects[0]
finetune=finetunes[0]
for percent in percents:
    opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
    cfg.merge_from_list(opts)
    corr_path=Path(str(create_dir_name(cfg,subj))+'/lightning_logs/version_%d/predictions/'%args.version)
    this_corr=torch.load(str(corr_path) + '/corr_combined.pt')
    corrs_mean.append(this_corr[this_corr>corr_cut].mean())
    corrs_std.append(this_corr[this_corr>corr_cut].std())

import matplotlib.pyplot as plt
plt.plot(percents,corrs_mean)
plt.savefig('/home/u2hussai/scratch/test.png')


#make the xmas tree plot
