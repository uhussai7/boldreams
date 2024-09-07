import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from dreams.objectives import roi
from torchvision.transforms import GaussianBlur,Compose,RandomAffine
from lucent.optvis import render, param, objectives
from torch.optim import Adam,SGD


ext='.yaml'
backbones=['alexnet','clip']
rois=['roi_floc-places','roi_floc-bodies','roi_floc-faces']

for backbone in backbones:

    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'%s.yaml'%backbone)
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
    nsd_data=NsdData([1])
    nsd_data.load_preprocessed(cfg.BACKBONE.INPUT_SIZE)    #handle text or not for data_loaders

    #handle text or not for data_loaders
    if cfg.BACKBONE.TEXT == True:
        import clip
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE,text=True,tokenizer=clip.tokenize)
    else:
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE)

    #get the encoder
    #Nv=int(nsd_data.data[0]['Nv']) #number of voxels
    #enc=modules.LitEncoder(cfg,nsd_data.data_loaders_train[0])


    #load the checkpoint
    Nv=int(nsd_data.data[0]['Nv']) #number of voxels
    checkpoint_path=str(create_dir_name(cfg,1))+'/lightning_logs/version_%d/checkpoints/'%0
    checkpoints=os.listdir(checkpoint_path)
    print('These are the checkpoints',checkpoints)
    epochs=np.asarray([int(checkpoint.split('epoch=')[1].split('-')[0]) for checkpoint in checkpoints])
    max_epoch_ind=np.argsort(epochs)[-1]
    max_epoch=epochs[max_epoch_ind]
    resume_checkpoint=checkpoint_path+checkpoints[max_epoch_ind]
    print('Loading checkpoint:',resume_checkpoint)
    enc=modules.LitEncoder.load_from_checkpoint(resume_checkpoint,cfg=cfg,data_loader=nsd_data.data_loaders_train[0])

    #get the validation data loader
    #predictor = modules.EncoderTrainer(cfg,subj=1,max_epochs=cfg.TRAIN.MAX_EPOCHS)
    #predictions = predictor.predict(enc.cuda(),nsd_data.data_loaders_val_single)
    #predictions=torch.cat(predictions,dim=1)


    
    for roi_ in rois:
        #dreaming#Dreamer
        rois_=nsd_data.data[0]['roi_dic_combined'].item()
        dreamer=dream_wrapper(enc,rois_)
        ##optimzier
        #optimizer=lambda params: Adam(params,lr=5e-3)
        optimizer=lambda params: SGD(params,lr=2)
        ##initial image
        param_f = lambda: param.image(cfg.BACKBONE.INPUT_SIZE[0], fft=True, decorrelate=True,sd=0.01)
        #param_f = lambda: ref_image(img)
        ##transforms
        jitter_only= [RandomAffine(1,translate=[0.1,0.1],scale=[0.5,0.9], fill=0.0)]
        #jitter_only= [RandomAffine(1,translate=[0.1,0.1],scale=[1,1], fill=0.0)]
        ##objective    
        obj = roi(roi_) #- 1e-5* diversity("roi_ffa") #+ 1.2*roi_mean_target(['roi_v1'],torch.tensor([-2]).cuda())
        ##rendering and plotting
        _=render.render_vis(dreamer.cuda().eval(),obj,param_f=param_f,transforms=jitter_only,
                        optimizer=optimizer,fixed_image_size=cfg.BACKBONE.INPUT_SIZE[0],thresholds=(1024,),show_image=False)
        #[plt.subplots()[1].imshow(_[-1][0]) for i in range(0,1)]
        #plt.savefig('/home/u2hussai/scratch/dream.png')

        #let go of encoder 
        #del enc

        #load clip
        cfg=get_cfg_defaults()
        cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'clip.yaml')

        clp=torch.load(cfg.PATHS.BACKBONE_FILES+cfg.BACKBONE.FILE)

        #pass image through the img part
        from torchvision.transforms import Resize

        img=_[0][0]
        img=torch.from_numpy(img)
        img=img.moveaxis(-1,0).unsqueeze(0)
        img=Resize(cfg.BACKBONE.INPUT_SIZE)(img)
        img_embedding=clp.encode_image(img.cuda()).detach().cpu()


        #load the text embeddings of common words
        #text_embeddings=torch.load(cfg.PATHS.NSD_TEXT + '/corpus_encode/' + 'encoded_corpus.pt')
        text_embeddings=torch.load(cfg.PATHS.NSD_TEXT + '/corpus_encode/' + 'brown_nouns-embedding_N-%d.pt'%2000)
        overlap=torch.matmul(text_embeddings,img_embedding.T)
        inds=overlap[:,0].argsort(descending=True)

        img_embedding /= img_embedding.norm(dim=-1, keepdim=True)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        similarity = (100.0 * img_embedding @ text_embeddings.T).softmax(dim=-1)
        values, indices = similarity[0].topk(100)

        from nltk.corpus import words,brown
        from collections import Counter
        Ncommon=20000
        common_words=brown.words()
        words_with_pos=brown.tagged_words(tagset='universal')
        nouns = [word for word, pos in words_with_pos if pos == 'NOUN']
        common_nouns = [word for word, _ in Counter(nouns).most_common(Ncommon)]

        #make a figure
        fig,ax=plt.subplots(1,2)
        ax[0].axis('off')
        ax[0].imshow(_[0][0])

        N=20
        top_words=[common_nouns[indices[i]] for i in range(0,N)]
        top_values=[values[i] for i in range(0,N)]
        top_words.reverse()
        top_values.reverse()
        ax[1].barh(top_words,top_values)
        plt.tight_layout()
        plt.savefig('/home/u2hussai/scratch/%s-%s.png'%(backbone,roi_))


        # tokens=[]
        # for word in tqdm(common_words):
        #     tokens.append(clip.tokenize(word))
        # tokens=torch.cat(tokens,0)

        # fmri_out=[]
        # batch_size=10
        # for i in tqdm(range(0,len(tokens),batch_size)):
        #     fmri_out.append(enc(tokens[i:i+batch_size].cuda().unsqueeze(1)).detach().cpu())