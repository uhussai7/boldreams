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

def get_ecc_roi(cfg,subj,x,y,z): #the eccentricity data has been thresholded, this needs to reloaded
    prf=nib.load(os.path.join(cfg.PATHS.MASK_ROOT,'subj%02d'%subj,cfg.FMRI.FUNC_RES,'roi','prf-visualrois.nii.gz')).get_fdata()[x,y,z]
    prf_ecc=nib.load(os.path.join(cfg.PATHS.MASK_ROOT,'subj%02d'%subj,cfg.FMRI.FUNC_RES,'roi','prf-eccrois.nii.gz')).get_fdata()[x,y,z]
    return prf,prf_ecc
        
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

def main():
    parser = argparse.ArgumentParser(
        description="Script for training encoders"
    )

    parser.add_argument(
        '-s','--subj',
        type=int,
        help='Integer id of subject, e.g. 1'
    )

    parser.add_argument(
        '-f','--finetune',
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
        default=False,
        help='Flag to toggle bacbone finetuning, True will finetune backbone'
    )

    parser.add_argument(
        '-p','--percent',
        type=int,
        default=100,
        help='Percentage of total filters per layer to extract for readout'
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

    parser.add_argument(
        '-r', '--roi',
        type=str,
        default=None,
        help='Roi for dream'
    )

    args=parser.parse_args()
    print(args)

    # parser.add_argument(
    #     '-v', '--version',
    #     type=int,
    #     default=0,
    #     help='If continuing from last checkpoint provide the version'
    # )

    #sort out the config file
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+args.config)
    args.percent=100 if args.percent is None else args.percent
    opts=["BACKBONE.FINETUNE",args.finetune,"BACKBONE.PERCENT_OF_CHANNELS",args.percent, "DREAMS.TRANSLATE", (0.05,0.05) ,"DREAMS.SCALE", (1,1) ]
    cfg.merge_from_list(opts) 
    cfg.freeze()
    print(cfg)

    subj=args.subj
    enc,cfg,nsd_data=load_model(cfg,subj=subj)

    #objectives=['default','diversity']

    prf,prf_ecc=get_ecc_roi(cfg,subj,nsd_data.data[0]['x'],nsd_data.data[0]['y'],nsd_data.data[0]['z'])

    #create an roi dic
    ecc_rois={}
    for i in range(0,int(prf_ecc.max()+1)):
        roi_name='ecc_%d'%i
        ecc_rois[roi_name]=prf_ecc==i

    print(ecc_rois.keys())

    objective_name='default'
    for roi_name in list(ecc_rois.keys()):

        path_name=str(os.path.join(cfg.PATHS.NSD_DREAMS,'subj%02d'%subj,cfg.BACKBONE.NAME,
                            'finetune-'+str(cfg.BACKBONE.FINETUNE),
                            'percent_channels-%d'%(int(cfg.BACKBONE.PERCENT_OF_CHANNELS)),
                            'objective-%s'%objective_name))
        print(path_name)
        out_path=Path(path_name)
        out_path.mkdir(parents=True,exist_ok=True)

        out_name='roi-%s'%roi_name + '_obj-%s'%objective_name + '.npy'

        #check if file exists:
        if Path(os.path.join(out_path,out_name)).exists():
            print('Dream file already exists')
            continue

        obj=choose_objective(roi_name,objective_name)
        dreams=dreaming_setup(enc,cfg,nsd_data,roi_name,ecc_rois,obj)    

        print(dreams[0].shape)
        print(os.path.join(out_path,out_name))
        np.save(os.path.join(out_path,out_name),dreams[0])


if __name__ == "__main__":
    main()