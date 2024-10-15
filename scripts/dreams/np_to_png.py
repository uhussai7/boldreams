import numpy as np
from configs.config import get_cfg_defaults
import os
import matplotlib.pyplot as plt
import torch

class model_params:
    def __init__(self,name,percents,finetune):
        self.name=name
        self.percents=percents
        self.finetune=finetune

models=[model_params('alexnet',[25,100],False),
        model_params('alexnet',[10,100],True),
        model_params('vgg11',[5,100],False),
        model_params('vgg11',[1,100],True),
        model_params('clip',[1,100],False)]

cfg=get_cfg_defaults()
objectives=['default','diversity']
subj=7
for model in models:
    for percent in model.percents:
        for objective in objectives:
            img_path=os.path.join(str(cfg.PATHS.NSD_DREAMS),'subj%02d'%subj,
                    str(model.name),'finetune-%s'%str(model.finetune),
                    'percent_channels-%d'%percent,'objective-%s'%str(objective))
            files=os.listdir(img_path)
            for file in files:
                img=torch.from_numpy(np.load(img_path+'/'+file))
                print(file)
                fig,axs=plt.subplots(1,4)
                for a,ax in enumerate(axs):
                    ax.imshow(img[a])
                    ax.axis('off')
                    print(img_path+"/"+file.split('.npy')[0]+'.png')
                    plt.savefig(img_path+"/"+file.split('.npy')[0]+'.png')
                plt.close()

                