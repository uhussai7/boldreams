import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import modules
from configs.config import get_cfg_defaults
from nsdhandling.core import NsdData
import numpy as np
import argparse
import os
from models.utils import create_dir_name
from pathlib import Path
import matplotlib.pyplot as plt
import torch

def compute_corr(preds):
    corr=torch.zeros(preds.shape[-1])
    for v in range(0,len(corr)):
        corr[v]=torch.corrcoef(preds[:,:,v])[0,1]
    corr[torch.isnan(corr)]=0
    return corr

def plot_save(corr,file_path,title='Histogram'):
    fig,ax=plt.subplots()
    ax.hist(corr,100)
    ax.set_title(title)
    plt.savefig(file_path)

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


    #load the checkpoint
    checkpoint_path=str(create_dir_name(cfg,args.subj))+'/lightning_logs/version_%d/checkpoints/'%args.version
    checkpoints=os.listdir(checkpoint_path)
    print('These are the checkpoints',checkpoints)
    epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
    max_epoch_ind=np.argsort(epochs)[-1]
    max_epoch=epochs[max_epoch_ind]
    resume_checkpoint=checkpoint_path+checkpoints[max_epoch_ind]
    print('Loading checkpoint:',resume_checkpoint)
    enc=modules.LitEncoder.load_from_checkpoint(resume_checkpoint,cfg=cfg,data_loader=nsd_data.data_loaders_train[0])

    #save the predictions and a histogram plot
    #handle the path
    predictions_path=Path(str(create_dir_name(cfg,args.subj))+'/lightning_logs/version_%d/predictions/'%args.version)
    predictions_path.mkdir(exist_ok=True)


    #get the validation data loader
    predictor = modules.EncoderTrainer(cfg,subj=args.subj,max_epochs=cfg.TRAIN.MAX_EPOCHS)

    #single
    predictions = predictor.predict(enc,nsd_data.data_loaders_val_single)
    predictions_single=torch.cat(predictions,dim=1)
    corr=compute_corr(predictions_single)
    plot_save(corr,str(predictions_path)+'/corr_single_hist.png','single')
    torch.save(corr,str(predictions_path)+'/corr_single.pt')

    #multi
    predictions = predictor.predict(enc,nsd_data.data_loaders_val_multi)
    predictions_multi=torch.cat(predictions,dim=1)
    corr=compute_corr(predictions_multi)
    plot_save(corr,str(predictions_path)+'/corr_multi_hist.png','multi')
    torch.save(corr,str(predictions_path)+'/corr_multi.pt')

    #combined
    corr=compute_corr(torch.cat([predictions_single,predictions_multi],dim=1))
    plot_save(corr,str(predictions_path)+'/corr_combined_hist.png','combined')
    torch.save(corr,str(predictions_path)+'/corr_combined.pt')


if __name__ == "__main__":
    main()

    # #compute the correlation
    # # corr=torch.zeros(predictions.shape[-1])
    # # for v in range(0,len(corr)):
    # #     corr[v]=torch.corrcoef(predictions[:,:,v])[0,1]
    
    # #compute the single correlations
    

    # fig,ax=plt.subplots()
    # ax.hist(corr,100)    
    # ax.set_title('single')
    # plt.savefig(str(predictions_path)+'/corr_single_hist.png')

    # #compute the multi predictions and correlations
    # predictions = predictor.predict(enc,nsd_data.data_loaders_val_multi)
    # predictions=torch.cat(predictions,dim=1)
    
    # corr=compute_corr(predictions)
    # torch.save(corr,str(predictions_path)+'/corr_multi.pt')

    # fig,ax=plt.subplots()
    # ax.hist(corr,100)
    # ax.set_title('multi')
    # plt.savefig(str(predictions_path)+'/corr_multi_hist.png')














# import sys
# import os 
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from models import modules
# from configs.config import get_cfg_defaults
# from nsdhandling.core import NsdData
# import numpy as np
# import argparse
# import os
# from models.utils import create_dir_name
# from pathlib import Path
# import matplotlib.pyplot as plt
# import torch


# def compute_corr(preds):
#     corr=torch.zeros(preds.shape[-1])
#     for v in range(0,len(corr)):
#         corr[v]=torch.corrcoef(preds[:,:,v])[0,1]
#     corr[torch.isnan(corr)]=0
#     return corr

# def make_predictions(subj,cfg,nsd_data,val_dataloader,finetune,percent,version):
#     opts=["BACKBONE.FINETUNE",finetune,"BACKBONE.PERCENT_OF_CHANNELS",percent]
#     cfg.merge_from_list(opts)
#     cfg.freeze()
#     print(cfg)

#     #load the checkpoint
#     checkpoint_path=str(create_dir_name(cfg,subj))+'/lightning_logs/version_%d/checkpoints/'%version
#     checkpoints=os.listdir(checkpoint_path)
#     print('These are the checkpoints',checkpoints)
#     epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
#     max_epoch_ind=np.argsort(epochs)[-1]
#     max_epoch=epochs[max_epoch_ind]
#     resume_checkpoint=checkpoint_path+checkpoints[max_epoch_ind]
#     print('Loading checkpoint:',resume_checkpoint)
#     enc=modules.LitEncoder.load_from_checkpoint(resume_checkpoint,cfg=cfg,data_loader=nsd_data.data_loaders_train[0])
    
#     predictions_path=Path(str(create_dir_name(cfg,subj))+'/lightning_logs/version_%d/predictions/'%version)
#     predictions_path.mkdir(exist_ok=True)

#     #get the validation data loader
#     predictor = modules.EncoderTrainer(cfg,subj=subj,max_epochs=cfg.TRAIN.MAX_EPOCHS)
#     predictions = predictor.predict(enc,val_dataloader)
#     predictions=torch.cat(predictions,dim=1)
    
#     #compute the single correlations
#     corr=compute_corr(predictions)

#     del enc
#     del predictor
#     del val_dataloader

#     return corr, predictions, predictions_path


# def main():
#     parser = argparse.ArgumentParser(
#         description="Script for training encoders"
#     )

#     parser.add_argument(
#         '-s','--subj',
#         type=int,
#         help='Integer id of subject, e.g. 1'
#     )

#     parser.add_argument(
#         '-f','--finetune',
#         type=lambda x: (str(x).lower() in ['true', '1', 'yes']), required=True,
#         help='Flag to toggle bacbone finetuning, True will finetune backbone'
#     )

#     parser.add_argument(
#         '-p','--percent',
#         type=int,
#         help='Percentage of total filters per layer to extract for readout'
#     )

#     parser.add_argument(
#         '-c','--config',
#         type=str,
#         help='Config file name, if not done, please define config folder as environment variable named NSD_CONFIG_PATH'
#     )

#     parser.add_argument(
#         '-v', '--version',
#         type=int,
#         default=0,
#         help='If continuing from last checkpoint provide the version'
#     )

#     args=parser.parse_args()
#     print(args)

#     #sort out the config file
#     cfg=get_cfg_defaults()
#     cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+args.config)
#     args.percent=[1, 5, 10, 15, 20, 25, 50, 75, 100] if args.percent is None else [args.percent]

#     for percent in args.percent:
#         #save the predictions and a histogram plot
#         #handle the path

#         #load the subject data
#         nsd_data=NsdData([args.subj])
#         nsd_data.load_preprocessed(cfg.BACKBONE.INPUT_SIZE)

#         #handle text or not for data_loaders
#         if cfg.BACKBONE.TEXT == True:
#             import clip
#             nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE,text=True,tokenizer=clip.tokenize)
#         else:
#             nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE)

#         corr,predictions_single,predictions_path=make_predictions(args.subj,cfg, nsd_data,nsd_data.data_loaders_val_single,
#                                                            args.finetune,percent,args.version)
#         torch.save(corr,str(predictions_path)+'/corr_single.pt')
#         fig,ax=plt.subplots()
#         ax.hist(corr,100)    
#         ax.set_title('single')
#         plt.savefig(str(predictions_path)+'/corr_single_hist.png')
#         print(predictions_single.shape)
#         del corr


#         #compute the multi predictions and correlations
#         corr,predictions_multi,predictions_path=make_predictions(args.subj,cfg, nsd_data,nsd_data.data_loaders_val_multi,
#                                                            args.finetune,percent,args.version)
#         corr=compute_corr(predictions_multi)
#         torch.save(corr,str(predictions_path)+'/corr_multi.pt')
#         fig,ax=plt.subplots()
#         ax.hist(corr,100)
#         ax.set_title('multi')
#         plt.savefig(str(predictions_path)+'/corr_multi_hist.png')
#         print(predictions_multi.shape)
#         del corr

#         corr=compute_corr(torch.cat([predictions_single,predictions_multi],dim=1))
#         fig,ax=plt.subplots()
#         ax.hist(corr,100)
#         ax.set_title('combined')
#         plt.savefig(str(predictions_path)+'/corr_combined_hist.png')
#         torch.save(corr,str(predictions_path)+'/corr_combined.pt')

#         del corr
#         del predictions_single
#         del predictions_multi



#     # #get the trainer
#     # #handle version, if specified
#     # if args.version is not None:
#     #     print('Checking version provided')
#     #     checkpoint_path=str(create_dir_name(cfg,args.subj))+'/lightning_logs/version_%d/checkpoints/'%args.version
#     #     checkpoints=os.listdir(checkpoint_path)
#     #     print('These are the checkpoints',checkpoints)
#     #     epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
#     #     max_epoch_ind=np.argsort(epochs)[-1]
#     #     max_epoch=epochs[max_epoch_ind]
#     #     if max_epoch<cfg.TRAIN.MAX_EPOCHS-1:
#     #         resume_checkpoint=checkpoint_path+checkpoints[max_epoch_ind]
#     #         print('Resuming from',resume_checkpoint)
#     #         trainer = modules.EncoderTrainer(cfg,subj=args.subj,
#     #                                          max_epochs=cfg.TRAIN.MAX_EPOCHS)
#     #         trainer.fit(model=enc.cuda(), train_dataloaders=enc.encoder.data_loader,
#     #                     ckpt_path=resume_checkpoint)

#     # else: #this will create new version
#     #     trainer = modules.EncoderTrainer(cfg,subj=args.subj,max_epochs=cfg.TRAIN.MAX_EPOCHS)
#     #     trainer.fit(model=enc.cuda(), train_dataloaders=enc.encoder.data_loader)

# if __name__ == "__main__":
#     main()