import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.config import get_cfg_defaults
import argparse
import torch
from pathlib import Path
from models.utils import create_dir_name
import smplotlib
import numpy as np
import matplotlib.pyplot as plt

#lets make the alexnet accuract plot
# parser = argparse.ArgumentParser(
#         description="Script for training encoders"
#     )

# parser.add_argument(
#     '-c','--config',
#     type=str,
#     help='Config file name, if not done, please define config folder as environment variable named NSD_CONFIG_PATH'
# )


# parser.add_argument(
#     '-v', '--version',
#     type=int,
#     default=0,
#     help='If continuing from last checkpoint provide the version'
# )




#some params
backbones = ('alexnet.yaml','vgg11.yaml','clip.yaml')
percents=(1, 5, 10, 15, 20, 25, 50, 75, 100)
subjects=(1,2,5,7)
subject_colors=('b','g','r','c')
finetunes=(False,True)
corr_cut=-100

fig=plt.figure(figsize=(15,15))
gs=fig.add_gridspec(3,3,wspace=0,hspace=0)


for b,backbone in enumerate(backbones):
    for f,finetune in enumerate(finetunes):
        ax = fig.add_subplot(gs[f,b])
        for s,subject in enumerate(subjects):
            corrs_mean = []
            corrs_std = []
            cfg=get_cfg_defaults()
            cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+backbone)
            for percent in percents:
                opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
                cfg.merge_from_list(opts)
                corr_path=Path(str(create_dir_name(cfg,subject))+'/lightning_logs/version_%d/predictions/'%0)
                this_corr=torch.load(str(corr_path) + '/corr_combined.pt')
                #this_corr[torch.isnan(this_corr)]=0
                print(this_corr[this_corr>corr_cut].mean())
                corrs_mean.append(this_corr[this_corr>corr_cut].mean())
                corrs_std.append(this_corr[this_corr>corr_cut].std())
            #corrs_mean=np.asarray(corrs_mean)
            print(corrs_mean)
            ax.plot(percents,corrs_mean,subject_colors[s])
        ax.set_ylim([-0.02,0.24])



# #load
# import matplotlib.pyplot as plt
# plt.figure(figsize=(5,6))
# for s,subject in enumerate(subjects):
#     corrs_mean = []
#     corrs_std = []
#     #subj=subjects[1]
#     finetune=finetunes[1]
#     for percent in percents:
#         opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
#         cfg.merge_from_list(opts)
#         corr_path=Path(str(create_dir_name(cfg,subject))+'/lightning_logs/version_%d/predictions/'%args.version)
#         this_corr=torch.load(str(corr_path) + '/corr_combined.pt')
#         corrs_mean.append(this_corr[this_corr>corr_cut].mean())
#         corrs_std.append(this_corr[this_corr>corr_cut].std())

#     N=(this_corr>corr_cut).sum()
#     plt.plot(percents,corrs_mean,subject_colors[s])
#     plt.errorbar(percents,corrs_mean,np.asarray(corrs_std)/np.sqrt(N),np.asarray(corrs_std)/np.sqrt(N),subject_colors[s])
#     plt.title(str(cfg.BACKBONE.NAME) + ' ' + str(cfg.BACKBONE.FINETUNE))
#     plt.xlabel('% of filters per layer')
#     plt.ylabel('Mean Correlation')
plt.savefig('/home/u2hussai/projects/def-uludagk/u2hussai/accuracy-vs-percent_allsubs.png')#%subject)


#make the xmas tree plot
