import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dreams import wrappers
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
from torchvision.transforms import GaussianBlur,Compose,RandomAffine,Resize
from lucent.optvis import render, param, objectives
from torch.optim import Adam,SGD
from tqdm import tqdm
from pathlib import Path
import smplotlib
import matplotlib.gridspec as gridspec


def load_mask(cfg=get_cfg_defaults(),subj=1,backbone='alexnet',percent='100',finetune='True',roi_name='floc-faces',img_id=0):
    return np.load(os.path.join(cfg.PATHS.NSD_PLOTS,'attention_maps' ,'subj%02d'%subj, backbone, 'finetune-%s'%str(finetune),
                                'percent_channels-%d'%int(percent),
                                roi_name,'mask_%d.npy'%img_id),allow_pickle=True)

#load masks and images
cfg=get_cfg_defaults()

#images
img_path=Path(cfg.PATHS.NSD_DREAMS + '/image_per_roi/' )
roi_names=os.listdir(str(img_path))
roi_names=np.unique([roi_name.split('_')[0] for roi_name in roi_names if 'ecc' not in roi_name])
print(roi_names)

img_ids={'V1':0, 'V2':0, 'V3':0, 'V3ab':0, 'VO':0,
        'IPS':0, 'LO':0, 'MST':0, 'MT':0, 'PHC':0,
        'hV4':0, 'floc-bodies':6,
        'floc-faces':8, 'floc-places':0, 'floc-words':0}

class model_params:
    def __init__(self,name,percents,finetune):
        self.name=name
        self.percents=percents
        self.finetune=finetune
        self.input_size=self.get_input_size()

    def load_mask(self,subj,roi_name,img_id):
        masks=[]
        for percent in self.percents:
            mask=torch.from_numpy(np.load(os.path.join(cfg.PATHS.NSD_PLOTS,'attention_maps' ,'subj%02d'%subj, self.name, 'finetune-%s'%str(self.finetune),
                                'percent_channels-%d'%int(percent),
                                roi_name,'mask_%d.npy'%img_id),allow_pickle=True))
            masks.append(self.mask_processing(mask))
        return masks

    def get_input_size(self):
        cfg=self.get_config()
        return cfg.BACKBONE.INPUT_SIZE

    def get_config(self):
        cfg=get_cfg_defaults()
        cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+self.name +'.yaml')
        opts=["BACKBONE.FINETUNE",self.finetune,"BACKBONE.PERCENT_OF_CHANNELS",self.percents[0]]
        cfg.merge_from_list(opts)
        return cfg
    
    def mask_processing(self,mask):
        mask=mask.abs().mean(0)
        mask=mask/mask.max()
        mask[mask<0.1]=0
        mask=GaussianBlur(21,sigma=(1.2,1.2))(mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        mask=np.clip(1.0*mask/mask.max(),0,1)
        #mask[mask<0.275]=0
        return mask.moveaxis(0,-1)
    
def apply_mask(image, mask):
    # Assuming image is of shape (3, H, W) and mask is of shape (H, W)
    # Ensure image and mask are tensors and on the same device
    # if isinstance(image, np.ndarray):
    #     image = torch.from_numpy(image).float() / 255.0
    # if isinstance(mask, np.ndarray):
    #     mask = torch.from_numpy(mask).float()

    # Convert to grayscale by averaging across RGB channels (or use more sophisticated conversion if needed)
    grayscale_image = 1.0*image.mean(dim=0, keepdim=True).repeat(3, 1, 1)

    # Brighten the color image by a factor (you can adjust this factor)
    bright_image = 1.00*image#torch.clamp(image * 2.5, 0, 1)

    # Reshape the mask to match the dimensions of the image (add channel dimension)
    mask = Resize(image.shape[-2:])(mask.T.unsqueeze(0).unsqueeze(0))[0]

    print(mask.shape)
    print(image.shape)

    # Blend the grayscale and brightened color image based on the mask
    output_image =2.50*bright_image*mask #+ 0.9*mask #+  grayscale_image * (1 - mask) 

    print(output_image.shape)

    return output_image


models=[model_params('alexnet',[25,100],False),
        model_params('alexnet',[10,100],True),
        model_params('vgg11',[5,100],False),
        model_params('vgg11',[1,100],True),
        model_params('clip',[1,100],False)]

roi_names=['V1','V2','V3','hV4','floc-faces','floc-bodies','floc-places','floc-words']
# roi_name='floc-places'#roi_names[-1]
# #img_id=1
# model_id=3

# masks=models[model_id].load_mask(1,roi_name,img_id)

# fig,ax=plt.subplots(1,2)
# ax.flatten()[0].imshow(test_img.moveaxis(0,-1))

# #ax[1].imshow(test_img.moveaxis(0,-1).detach().cpu().numpy())
# ax[1].imshow(apply_mask(test_img,masks[0]).moveaxis(0,-1))

# plt.savefig(str(cfg.PATHS.NSD_PLOTS) + './test.png')


#so we have only coloumn for image then each backbone and then fine tuine of and on and percents thats (10) models so 1 colums
# roi_name='V1'#roi_names[-1]
# img_id=8


# rows=len(roi_names)
# cols=6
# fig,ax_=plt.subplots(rows,cols,figsize=(cols,rows))

# for r,roi_name in enumerate(roi_names):
#     test_img=torch.from_numpy(np.load(str(img_path)+ '/%s_%d_img.npy'%(roi_name,img_id)))
#     ax=ax_.T[:,r].flatten()
#     ax[0].imshow(np.flipud(test_img.moveaxis(0,-1)))
#     ax[0].axis('off')

#     i=1
#     for model in models:
#         grid=gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=ax[i].get_subplotspec(), wspace=0, hspace=0)
#         masks=model.load_mask(7,roi_name,img_id)
#         for m,mask in enumerate(masks):
#             ax1=plt.Subplot(fig,grid[m])
#             ax1.imshow(np.flipud(apply_mask(test_img,mask).moveaxis(0,-1)))
#             ax1.axis('off')
#     i+=1

# plt.tight_layout()
# plt.savefig(str(cfg.PATHS.NSD_PLOTS) + './test.png')


rows = len(roi_names)
cols = 6
fig, ax_ = plt.subplots(rows, cols, figsize=(cols*2, rows),gridspec_kw={'wspace': 0.03, 'hspace': 0.015},dpi=1200)
subj=1

for r, roi_name in enumerate(roi_names):
    img_id=img_ids[roi_name]
    test_img = torch.from_numpy(np.load(str(img_path) + '/%s_%d_img.npy' % (roi_name, img_id)))
    
    # Accessing the axes
    # ax = ax_.T[:, r].flatten()
    # ax[0].imshow(np.flipud(test_img.moveaxis(0, -1)))
    # ax[0].axis('off')

    ax_[r, 0].axis('off')
    grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_[r, 0].get_subplotspec(), wspace=0, hspace=0)
    ax1 = fig.add_subplot(grid[1])
    ax1.imshow(np.flipud(test_img.moveaxis(0,-1)))
    ax1.axis('off')

    roi_name_=roi_name.split('-')
    if len(roi_name_)>1:
        roi_name_=roi_name_[1]
    else:
        roi_name_=roi_name
    fig.text(0.45, 0.5, '%s'%roi_name_, fontsize=12, ha='center', va='center',transform=ax_[r,0].transAxes,rotation=90)

    i = 1
    for model in models:
        # Creating gridspec for the current axes
        ax_[r, i].axis('off')
        grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_[r, i].get_subplotspec(), wspace=0.015, hspace=5)
        
        masks = model.load_mask(subj, roi_name, img_id)
        for m, mask in enumerate(masks):
            ax1 = fig.add_subplot(grid[m])
            ax1.imshow(np.flipud(apply_mask(test_img, mask).moveaxis(0, -1)))
            ax1.axis('off')
            if r==0:
                fig.text(0.25, 1.3, '%s'%((model.name).capitalize()), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
                fig.text(0.75, 1.3, '%s'%((model.name).capitalize()), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
                fig.text(0.25, 1.1, '%d-%s'%(model.percents[0],str(model.finetune)), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
                fig.text(0.75, 1.1, '%d-%s'%(model.percents[1],str(model.finetune)), fontsize=12, ha='center', va='center',transform=ax_[0,i].transAxes)
        i += 1

plt.tight_layout()
delta=1/12
fig.text(0.75, 1.1, 'Input', fontsize=12, ha='center', va='center',transform=ax_[0,0].transAxes)
# plt.savefig(str(cfg.PATHS.NSD_PLOTS) + './test.png')

file_path=Path(os.path.join(cfg.PATHS.NSD_PLOTS,'attention_maps','subj%02d'%subj))#,plot_name+'.png'))
file_path.mkdir(parents=True,exist_ok=True)
plt.savefig(str(file_path)+'/' + 'plot.png' + '',dpi=800)

print(1)
# ax.imshow(test)