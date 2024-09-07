import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import modules
from configs.config import get_cfg_defaults
from nsdhandling.core import NsdData
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from models.utils import create_dir_name
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
import torch 

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
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), required=True,
        help='Flag to toggle bacbone finetuning, True will finetune backbone'
    )

    parser.add_argument(
        '-p','--percent',
        type=int,
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

    args=parser.parse_args()
    print(args)

    #sort out the config file
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+args.config)
    args.percent=100 if args.percent is None else args.percent
    opts=["BACKBONE.FINETUNE",args.finetune,"BACKBONE.PERCENT_OF_CHANNELS",args.percent]
    cfg.merge_from_list(opts)
    cfg.freeze()
    print(cfg)

    #handle devices
    accelerator='cpu'
    N_devices=1
    strategy='ddp_find_unused_parameters_true'
    if torch.cuda.is_available():
        accelerator='gpu'
        N_devices=int(torch.cuda.device_count())

    #load the subject data
    nsd_data=NsdData([args.subj])
    nsd_data.load_preprocessed(cfg.BACKBONE.INPUT_SIZE)
    
    #handle text or not for data_loaders
    if cfg.BACKBONE.TEXT == True:
        import clip
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE,text=True,tokenizer=clip.tokenize)
    else:
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE)

    #get the encoder
    Nv=int(nsd_data.data[0]['Nv']) #number of voxels
    enc=modules.LitEncoder(cfg,nsd_data.data_loaders_train[0])

    #handle the versioning correctly
    version=args.version
    root_dir=str(create_dir_name(cfg,args.subj))
    checkpoint_path=str(create_dir_name(cfg,args.subj))+'/lightning_logs/version_%d/checkpoints/'%version
    resume_checkpoint=None
    if version is None:
        trainer = modules.EncoderTrainer(cfg,subj=args.subj,max_epochs=cfg.TRAIN.MAX_EPOCHS,logger=CSVLogger(root_dir),accelerator=accelerator,devices=N_devices,strategy=strategy)
        trainer.fit(model=enc, train_dataloaders=enc.encoder.data_loader)
    else:
        if os.path.exists(checkpoint_path):
            checkpoints=os.listdir(checkpoint_path)
            if len(checkpoints)>0:
                print('These are the checkpoints in the version provided...',checkpoints)
                epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
                max_epoch_ind=np.argsort(epochs)[-1]
                max_epoch=epochs[max_epoch_ind]
                resume_checkpoint=checkpoint_path+checkpoints[max_epoch_ind]
            else:
                print('No checkpoints found, will put checkpoints from training in the version provided')
        else:
            print('Version is provided but version path does not exist. I will make it.')
            Path(checkpoint_path).mkdir(parents=True,exist_ok=True)
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path)    
        logger=CSVLogger(save_dir=root_dir,version=version)
        trainer = modules.EncoderTrainer(cfg,subj=args.subj,max_epochs=cfg.TRAIN.MAX_EPOCHS,logger=logger,callbacks=[checkpoint_callback],accelerator=accelerator,devices=N_devices,strategy=strategy)
        trainer.fit(model=enc, train_dataloaders=enc.encoder.data_loader,ckpt_path=resume_checkpoint)

if __name__ == "__main__":
    main()