import clip
from nltk.corpus import words,brown
from tqdm import tqdm
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config import get_cfg_defaults
from pathlib import Path
import torch
from collections import Counter


#lets use brown and get the N most used verbs and nouns and pass it through the transforme


def tokenize(words):
    print('Tokenizing words..')
    tokens=[]
    for word in tqdm(words):
        tokens.append(clip.tokenize(word))
    tokens=torch.cat(tokens,0)
    return tokens

def model_pass(model,tokens,batch_size=10,device='cuda'):
    print('Passing through model')   
    token_outs=[]
    for i in tqdm(range(0,len(tokens),batch_size)):
        token=tokens[i:i+batch_size]
        token_outs.append(model.encode_text(token.to(device)).detach().cpu())
    token_outs=torch.cat(token_outs,dim=0)
    return token_outs

def with_brown(model,out_path,Ncommon=20000,device='cuda'):

    words_with_pos=brown.tagged_words(tagset='universal')

    nouns = [word for word, pos in words_with_pos if pos == 'NOUN']
    verbs = [word for word, pos in words_with_pos if pos == 'VERB']

    # Get the most common nouns and verbs
    common_nouns = [word for word, _ in Counter(nouns).most_common(Ncommon)]
    common_verbs = [word for word, _ in Counter(verbs).most_common(Ncommon)]

    tokens_nouns=tokenize(common_nouns)
    tokens_verbs=tokenize(common_verbs)
    
    nouns_embedding=model_pass(model,tokens_nouns)
    verbs_embedding=model_pass(model,tokens_verbs)

    torch.save(nouns_embedding,str(out_path)+'/brown_nouns-embedding_N-%d.pt'%(Ncommon))
    torch.save(verbs_embedding,str(out_path)+'/brown_verbs-embedding_N-%d.pt'%(Ncommon))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #config
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'/clip.yaml')
    cfg.freeze()
    print(cfg)

    #make output dir
    out_path=Path(cfg.PATHS.NSD_TEXT + '/corpus_encode/')
    out_path.mkdir(parents=True,exist_ok=True)

    #get the model
    model=torch.load(cfg.PATHS.BACKBONE_FILES+cfg.BACKBONE.FILE).to(device)

    with_brown(model,out_path,device=device)

if __name__ == "__main__":
    main()