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
import smplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib import cm


# plt.rcParams.update({
#     #"text.usetex": True
#     # "font.family": "serif",
#     # "font.serif": ["Palatino"],
#     # "font.size": 22,
# })

def xmas_plot(corr1,corr2,ax,xmin=-0.24,xmax=0.24,ymin=0,ymax=0.78,remove_ylabels=False,remove_xlabels=False,max_value=75,percent=1,finetune=True):
    #print("using %d data points"%len(corr1))
    
    #if percent==1:
    ax.text(0.05,0.8, str(finetune)[0], transform=ax.transAxes, color='grey', fontsize='small')
    if remove_ylabels:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.set_ylabel(r'max$(\rho_i$, $\rho_{100})$',fontsize='small')

    if remove_xlabels:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
    else:
        ax.tick_params(axis='x', which='major', pad=1)
        ax.set_xlabel(r'$\rho_{%d}$ - $\rho_{100}$'%percent,fontsize='small')
        ax.xaxis.set_label_coords(0.5, -0.1) 

    aspect='auto'#(xmax-xmin)/(ymax-ymin)

    #zz=rand_img=np.zeros([100,100])*max_value#np.random.rand(100,100)*max_value

    #get x,y values
    y=np.asarray([corr1, corr2]).max(0)
    x=corr1-corr2

    #make a grid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    #make the kernel
    values = np.vstack([x, y])
    kde = gaussian_kde(values)
    zz = np.reshape(kde(positions).T, xx.shape)
    
    #evaluate
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions).T, xx.shape)
    ax.imshow(zz.T/max_value, extent=[xmin, xmax, ymin, ymax],cmap='Greys_r',aspect=aspect)

    print(zz.max())

    ax.tick_params(axis='x', labelsize=6.3) 
    ax.tick_params(axis='both', which='major', color='grey')  # Change major ticks
    ax.tick_params(axis='both', which='minor', color='grey')
    ax.spines['bottom'].set_color('grey')  # X-axis line
    ax.spines['left'].set_color('grey') 
    ax.tick_params(axis='x', labelcolor='black')  # Keep x-axis tick label color as black
    ax.tick_params(axis='y', labelcolor='black')

    ax.axvline(0, color='red', linestyle='--', linewidth=0.8)

    #ax.hexbin(x,y,cmap='inferno', bins=30,mincnt=1)
    #ax.set_xlim([xmin,xmax])

def get_corr(backbone,subj,percent,finetune,version=0):
    #return 1
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'/'+backbone + '.yaml')
    opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
    cfg.merge_from_list(opts)
    cfg.freeze()
    if type(subj)==list:
        corr=[]
        for s in subj: #
            corr_path=Path(str(create_dir_name(cfg,s))+'/lightning_logs/version_%d/predictions/'%version)
            corr.append(torch.load(str(corr_path) + '/corr_combined.pt'))
        return torch.cat(corr)
    else:
        corr_path=Path(str(create_dir_name(cfg,subj))+'/lightning_logs/version_%d/predictions/'%version)
        return torch.load(str(corr_path) + '/corr_combined.pt')
    

def plot_combined():

    #alright lets make the final figure
    percents=(1, 5, 10, 15, 20, 25, 50, 75, 100)
    backbones = ('alexnet','vgg11','clip')
    backbone_titles=('AlexNet','Vgg11','RN50x4')
    finetunes=(False,True)

    max_percents=len(percents)-1

    W=1.7 #size of each tile
    spacing= 0.4 #spacing between each subplot
    width= W*(max_percents) 
    height= (2 * W * len(backbones)) + 5*spacing -W
    fig=plt.figure(figsize=(width, height))
    plt.gca().set_axis_off()
    outer_grid = gridspec.GridSpec(3,1,hspace=spacing,wspace=0)


    rand_img=np.zeros([100,100])#np.random.rand(100,100)

    for b,backbone in enumerate(backbones):
        #if backbone!='clip':
        inner_grid=gridspec.GridSpecFromSubplotSpec(2,max_percents,subplot_spec=outer_grid[b], wspace=0, hspace=0)

        outer_ax=plt.Subplot(fig, outer_grid[b])
        outer_ax.set_title('Backbone: %s'%backbone_titles[b])
        outer_ax.axis('off')
        fig.add_subplot(outer_ax)

        #else:
            #inner_grid=gridspec.GridSpecFromSubplotSpec(2,max_percents,subplot_spec=outer_grid[b], wspace=0, hspace=0)
        for p in range(max_percents):
            if p>0:
                remove_ylabels=True
            else:
                remove_ylabels=False
            ax = plt.Subplot(fig, inner_grid[0,p])
            
            if backbone!='clip':
                xmas_plot(get_corr(backbone,[1,2,5],percents[p],False),get_corr(backbone,[1,2,5],100,False),ax,remove_ylabels=remove_ylabels, remove_xlabels=True,percent=percents[p],finetune=False)
                fig.add_subplot(ax)
                ax = plt.Subplot(fig, inner_grid[1,p])
                #ax.imshow(rand_img,extent=[0, 1, 0, 1],aspect)
                xmas_plot(get_corr(backbone,[1,2,5],percents[p],True),get_corr(backbone,[1,2,5],100,True),ax,remove_ylabels=remove_ylabels, remove_xlabels=False,percent=percents[p],finetune=True)
                fig.add_subplot(ax)
            else:
                xmas_plot(get_corr(backbone,[1,2,5],percents[p],False),get_corr(backbone,[1,2,5],100,False),ax,remove_ylabels=remove_ylabels, remove_xlabels=False,percent=percents[p],finetune=False)
                fig.add_subplot(ax)

    norm = mcolors.Normalize(vmin=0, vmax=75)  # Adjust vmin and vmax according to your data
    sm = plt.cm.ScalarMappable(cmap=cm.Greys_r, norm=norm)
    sm.set_array([])  # Required for the color bar
    
    # Add color bar at the bottom
    cbar_ax = fig.add_axes([0.25, 0.13, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Number of voxels/bin')  # Add label to the color bar


    cfg=get_cfg_defaults()
    out_path=Path(cfg.PATHS.NSD_PLOTS + '/accuracy/')
    out_path.mkdir(parents=True,exist_ok=True)
    plt.savefig( str(out_path) + '/accuracy_xmas.png', dpi=800)#%subject)


    #plt.savefig('/home/u2hussai/projects/def-uludagk/u2hussai/test.png')
 

plot_combined()  


# gs=fig.add_gridspec(2,len(percents)-1,wspace=0,hspace=0)

# for p in range(0,len(percents)-1):
#     ax=fig.add_subplot(gs[0,p])
#     if p>0:
#         remove_ylabels=True
#     xmas_plot(get_corr('clip',[1,2,5],percents[p],False),get_corr('clip',[1,2,5],100,False),ax,remove_ylabels=remove_ylabels, remove_xlabels=True)
#     ax=fig.add_subplot(gs[1,p])
#     #xmas_plot(get_corr('clip_finetune',1,percents[p],True),get_corr('clip_finetune',1,100,True),ax,remove_ylabels=remove_ylabels, remove_xlabels=False)
# #lt.tight_layout()
# gs.update(wspace=0, hspace=0)
# plt.savefig('/home/u2hussai/projects/def-uludagk/u2hussai/test.png')




# #two rows per backbone except clip
# fig, ax =plt.subplots(3)
# for backbone in backbones:
