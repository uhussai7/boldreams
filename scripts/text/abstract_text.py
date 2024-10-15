import clip
from nltk.corpus import words,brown
from tqdm import tqdm
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs.config import get_cfg_defaults
from pathlib import Path
import torch
from collections import Counter
import numpy as np
from torchvision.transforms import GaussianBlur,Compose,RandomAffine,Resize
import inflect
import argparse
import matplotlib.pyplot as plt
import pickle

def with_brown(out_path,Ncommon=20000,device='cuda',N=1000):
    inflect_engine = inflect.engine()
    words_with_pos=brown.tagged_words(tagset='universal')

    #extract nounds and verbs
    nouns = [word for word, pos in words_with_pos if pos == 'NOUN']
    verbs = [word for word, pos in words_with_pos if pos == 'VERB']

    
    # Get the most common nouns and verbs
    common_nouns = [word.lower() for word, _ in Counter(nouns).most_common(Ncommon)]
    common_verbs = [word.lower() for word, _ in Counter(verbs).most_common(Ncommon)]

    #singularize
    singular_nouns = [inflect_engine.singular_noun(word) or word for word in common_nouns]
    singular_verbs = [inflect_engine.singular_noun(word) or word for word in common_verbs]

    noun_indices = {}
    unique_singular_nouns = []
    for i, word in enumerate(singular_nouns):
        if word not in noun_indices:
            noun_indices[word] = i  # Track the original index
            unique_singular_nouns.append(word)

    # Remove duplicates and keep the order of the first occurrence (for verbs)
    verb_indices = {}
    unique_singular_verbs = []
    for i, word in enumerate(singular_verbs):
        if word not in verb_indices:
            verb_indices[word] = i  # Track the original index
            unique_singular_verbs.append(word)

    # Filter embeddings to match the filtered nouns and verbs
    filtered_nouns_indices = [noun_indices[word] for word in unique_singular_nouns]
    filtered_verbs_indices = [verb_indices[word] for word in unique_singular_verbs]

    nouns_embedding=torch.load(str(out_path)+'/brown_nouns-embedding_N-%d.pt'%(Ncommon))[filtered_nouns_indices]
    verbs_embedding=torch.load(str(out_path)+'/brown_verbs-embedding_N-%d.pt'%(Ncommon))[filtered_verbs_indices]

    return  unique_singular_nouns[:N], nouns_embedding[:N], unique_singular_verbs[:N],verbs_embedding[:N]

#parser
def main():
    parser = argparse.ArgumentParser(
            description="Script for handling text for dreams"
        )

    parser.add_argument(
        '-s','--subj',
        type=int,
        default=1,
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
        default='clip.yaml',
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

    #config
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+args.config)
    args.percent=100 if args.percent is None else args.percent
    opts=["BACKBONE.FINETUNE",args.finetune,"BACKBONE.PERCENT_OF_CHANNELS",args.percent]
    cfg.merge_from_list(opts)
    cfg.freeze()
    print(cfg)


    #get clip
    #config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg1=get_cfg_defaults()
    cfg1.merge_from_file(cfg1.PATHS.NSD_CONFIG+'/clip.yaml')
    cfg1.freeze()
    print(cfg)
    model=torch.load(cfg1.PATHS.BACKBONE_FILES+cfg1.BACKBONE.FILE).to(device)

    #embeddings

    #get a dream
    
    objectives=['default','diversity']
    Nwords=[500,1000]

    for Nword in Nwords:
        nouns,nouns_embedding, verbs,verbs_embedding = with_brown(Path(cfg.PATHS.NSD_TEXT + '/corpus_encode/'),2000,Nword)
        for objective_name in objectives:
            in_path=Path(os.path.join(cfg.PATHS.NSD_DREAMS,'subj%02d'%args.subj,cfg.BACKBONE.NAME,
                                    'finetune-'+str(cfg.BACKBONE.FINETUNE),
                                    'percent_channels-%d'%(int(cfg.BACKBONE.PERCENT_OF_CHANNELS)),
                                    'objective-%s'%objective_name))
            roi_names = [roi_name.split('_')[0].split('roi-')[1] for roi_name in os.listdir(str(in_path)) if 'ecc' not in roi_name]
            for roi_name in roi_names:
                    in_name='roi-%s'%roi_name + '_obj-%s'%objective_name + '.npy'
                    dreams=torch.from_numpy(np.load(os.path.join(in_path,in_name))).moveaxis(-1,1)

                    #pass images
                    dreams=Resize(cfg1.BACKBONE.INPUT_SIZE)(dreams)
                    dream_embedding=model.encode_image(dreams.cuda()).detach().cpu()

                    dream_embedding /= dream_embedding.norm(dim=-1, keepdim=True)
                    nouns_embedding /= nouns_embedding.norm(dim=-1, keepdim=True)
                    verbs_embedding /= verbs_embedding.norm(dim=-1, keepdim=True)

                    nouns_similarity = (100.0 * dream_embedding @ nouns_embedding.T).softmax(dim=-1)
                    verbs_similarity = (100.0 * dream_embedding @ verbs_embedding.T).softmax(dim=-1)

                    out_dic={'nouns':nouns,
                            'nouns_similarity': nouns_similarity,
                            'verbs':verbs,
                            'verbs_similarity': verbs_similarity
                            }

                    out_path=Path(os.path.join(cfg.PATHS.NSD_TEXT, 'dreams', 'subj%02d'%args.subj,cfg.BACKBONE.NAME,
                                                'finetune-'+str(cfg.BACKBONE.FINETUNE),
                                                'percent_channels-%d'%(int(cfg.BACKBONE.PERCENT_OF_CHANNELS)),
                                                'objective-%s'%objective_name))
                    print(out_path)
                    out_path.mkdir(parents=True,exist_ok=True)
                    out_name='roi-%s'%roi_name + '_obj-%s'%objective_name + '_N-%d'%(Nword) + '.pkl'
                    with open(os.path.join(out_path,out_name),'wb') as f:
                        pickle.dump(out_dic,f)

if __name__ == "__main__":
    main()
    
# fig,ax=plt.subplots(1,4)
# for i in range(0,4):
#     values, indices = similarity[i].topk(10)
#     print([nouns[indice] for indice in indices])    
#     ax.flatten()[i].imshow(dreams[i].moveaxis(0,-1))
# plt.savefig(cfg.PATHS.NSD_PLOTS + 'test_%s.png'%roi_name)


#okay so should we do 2000 1000 and 500? might as well

#where do we put this? nsd dreams

#current structure is nsd_dreams/subj01/alexnet/finetune-False/percent_channels-100/objective/roi # okay so just put it where you load it
 

 #lets nsd_text/dreams/subj01/alexnet/finetune-False/percent_channels-100/objective/roi 