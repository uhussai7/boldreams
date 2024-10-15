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

#some params
backbones = ('alexnet.yaml','vgg11.yaml','clip.yaml')
backbone_titles=('AlexNet','Vgg11','RN50x4')
percents=(1, 5, 10, 15, 20, 25, 50, 75, 100)
subjects=(1,2,5,7)
subject_colors=('b','g','r','c')
subject_ids=(1,2,5,7)
finetunes=(False,True)
corr_cut=-100

#get default_cfg

#need to load roi info
def get_roi_dics(subs):
    cfg=get_cfg_defaults()
    roi_dic_combined_list=[]
    roi_dic_list=[]
    for s in subs:
        roi_dic_combined=np.load(cfg.PATHS.NSD_PREPROC + '/subj0%d/'%s + '/roi_dic_combined.npy',allow_pickle=True).all()
        roi_dic=np.load(cfg.PATHS.NSD_PREPROC + '/subj0%d/'%s + '/roi_dic.npy',allow_pickle=True).all()
        roi_dic_combined_list.append(roi_dic_combined)
        roi_dic_list.append(roi_dic)
    return roi_dic_combined_list,roi_dic_list

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

def corr_mean_over_subjects(roi,roi_dic_list,corr):
    corrs=[]
    for i in range(0,len(corr)):
        corrs.append(corr[i][roi_dic_list[i][roi]])
    corrs=torch.cat(corrs)
    return corrs.mean(), corrs.std()

#chose best model for each backbone + text and show correlation per ROI as bar chart
models=[{'backbone':'alexnet.yaml', 'finetune':False, 'percent': 25,'color':'b', 'label': 'AlexNet-False-25'},
        {'backbone':'alexnet.yaml', 'finetune':True, 'percent': 10,'color':'g','label':'AlexNet-True-10'},
        {'backbone':'vgg11.yaml', 'finetune':False, 'percent': 5,'color':'r','label':'Vgg11-False-5'},
        {'backbone':'vgg11.yaml', 'finetune':True, 'percent': 1,'color':'c','label':'Vgg11-True-1'},
        {'backbone':'clip.yaml', 'finetune':False, 'percent': 1,'color':'m','label':'RN50x4-False-1'},
        {'backbone':'clip_text.yaml', 'finetune':False, 'percent': 100, 'color':'k','label':'Clip-text-100'}
]

fig,ax=plt.subplots(figsize=(5,7))
subjs=[1,2,5]



roi_dic_list,_=get_roi_dics(subjs)
roi_names=list(roi_dic_list[0].keys())

roi_names=roi_names[:-1]
roi_names.reverse()

print(roi_names)

centers=np.arange(0,len(roi_names))*1.5
offsets=np.arange(0,len(models))
width=0.16667
offsets=np.arange(0,len(models))*width
offsets=offsets-offsets.max()/2
k=0

for r in range(0,len(roi_names)):
    custom_labels=[]
    custom_colors=[]
    for m,model in enumerate(models):
        #if m==0:
            #plt.legend()
        #model=models[0]
        corr=get_corr(subjs,model['backbone'],model['percent'],model['finetune'])
        corr_mean,corr_std=corr_mean_over_subjects(roi_names[r],roi_dic_list,corr)
        print(corr_mean)
        offset=width*k
        rects=ax.barh(centers[r]+offsets[len(offsets)-1-m],corr_mean.item(),width,color=model['color'],label=model['label'])
    #ax.bar_label(rects)
        k+=1
        custom_labels.append(model['label'])
        custom_colors.append(model['color'])


from matplotlib.ticker import FixedLocator

ax.set_yticks(centers)
ax.set_yticklabels(roi_names)

# Turn off minor ticks on the y-axis only
ax.yaxis.set_minor_locator(FixedLocator([]))

# Ensure that only the major ticks are displayed exactly at the specified locations on the y-axis
ax.yaxis.set_major_locator(FixedLocator(centers))
#ax.set_yticks(centers,roi_names)
custom_handles = [plt.Rectangle((0,0),1,1, color=color) for color in custom_colors]
plt.xlim([0,0.44])
plt.xlabel('Mean correlation')
plt.legend(custom_handles, custom_labels,loc='lower right',fontsize='xx-small')


cfg=get_cfg_defaults()
out_path=Path(cfg.PATHS.NSD_PLOTS + '/accuracy/')
out_path.mkdir(parents=True,exist_ok=True)
plt.savefig( str(out_path) + '/accuracy-roi-text.png',dpi=800)#%subject)


#plt.yticks(rotation=45)
# #can compare alexnet_finetune-on_percent= and text
# p=2
# b=0
# f=1
# cfg=get_cfg_defaults()
# cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+backbones[b])
# opts=["BACKBONE.FINETUNE",finetunes[f],"BACKBONE.PERCENT_OF_CHANNELS",percents[p]]
# cfg.merge_from_list(opts)
# print(cfg)
# corr_path=Path(str(create_dir_name(cfg,subject_ids[s]))+'/lightning_logs/version_%d/predictions/'%0)
# corr_alexnet=torch.load(str(corr_path) + '/corr_combined.pt')


# cfg=get_cfg_defaults()
# cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'clip_text.yaml')
# corr_path=Path(str(create_dir_name(cfg,subject_ids[s]))+'/lightning_logs/version_%d/predictions/'%0)
# corr_text=torch.load(str(corr_path) + '/corr_combined.pt')

# fig,ax=plt.subplots(figsize=(24,4))
# for roi_name,roi_filter in roi_dic_combined.items():
#     this_corr_alexnet=corr_alexnet[roi_filter]
#     this_corr_text=corr_text[roi_filter]
#     this_corr_alexnet_mean=this_corr_alexnet.mean()
#     this_corr_alexnet_std=this_corr_alexnet.std()
#     this_corr_text_mean=this_corr_text.mean()
#     this_corr_text_std=this_corr_text.std()
#     ax.plot(roi_name,this_corr_alexnet_mean,'rx')
#     ax.plot(roi_name,this_corr_text_mean,'bx')
# cfg=get_cfg_defaults()
# plt.savefig(str(cfg.PATHS.NSD_PLOTS) + '/test.png')