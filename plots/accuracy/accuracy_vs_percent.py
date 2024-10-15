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
from pathlib import Path

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
backbone_titles=('AlexNet','Vgg11','RN50x4')
percents=(1, 5, 10, 15, 20, 25, 50, 75, 100)
subjects=(1,2,5,7)
subject_colors=('b','g','r','c')
subject_ids=(1,2,5,7)
finetunes=(False,True)
corr_cut=-100

fig,ax_=plt.subplots(3,2,figsize=(9,10.5))
#gs=fig.add_gridspec(3,3,wspace=0,hspace=0)


for b,backbone in enumerate(backbones):
    for f,finetune in enumerate(finetunes):
        ax = ax_[b,f]#fig.add_subplot(gs[f,b])
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
                # print(this_corr[this_corr>corr_cut].mean())
                corrs_mean.append(this_corr[this_corr>corr_cut].mean())
                corrs_std.append(this_corr[this_corr>corr_cut].std())
            #corrs_mean=np.asarray(corrs_mean)
            # print(corrs_mean)
            ax.plot(percents,corrs_mean,subject_colors[s],label='subj %d'%subject_ids[s])
            N=(this_corr>corr_cut).sum()
            ax.errorbar(percents,corrs_mean,np.asarray(corrs_std)/np.sqrt(N),np.asarray(corrs_std)/np.sqrt(N),subject_colors[s])
        if backbone=='clip.yaml' and finetune == True:
            ax.legend(fontsize='x-small',loc='upper right') 
            ax.annotate('subjects overlap',(50,0),(50,0.04),fontsize='x-small',horizontalalignment='center', verticalalignment='bottom',
                        arrowprops=dict(arrowstyle='->',facecolor='black'))
        else:
            ax.legend(fontsize='x-small',loc='lower right') 
        ax.set_ylim([-0.02,0.29])
        ax.set_xlabel('% of filters/layer',fontsize='small')
        ax.set_ylabel('Mean correlation',fontsize='small')
        ax.set_title('Backbone: %s Finetune: %s'%(backbone_titles[b],str(finetune)))
plt.tight_layout()


cfg=get_cfg_defaults()
out_path=Path(cfg.PATHS.NSD_PLOTS + '/accuracy/')
out_path.mkdir(parents=True,exist_ok=True)
plt.savefig( str(out_path) + '/accuracy-vs-percent_allsubs.png',dpi=800)#%subject)

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


#make the xmas tree plot
