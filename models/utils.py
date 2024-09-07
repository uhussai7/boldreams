from collections import OrderedDict
import torch
from tqdm import tqdm
from torch import matmul
from pathlib import Path
import os

def create_dir_name(cfg,subj):
    out_path=Path(os.path.join(cfg.PATHS.NSD_ENCODER,'subj%02d'%subj,cfg.BACKBONE.NAME,
                          'finetune-'+str(cfg.BACKBONE.FINETUNE),
                          'percent_channels-%d'%(int(cfg.BACKBONE.PERCENT_OF_CHANNELS))))
    out_path.mkdir(parents=True,exist_ok=True)

    return str(out_path)


def channel_summer(sd):
    N=0
    for key in sd.keys():
        N+=len(sd[key])
    return N

def dic_mean_2d(dic,layers):
    for key in layers:
        dic[key]=dic[key].mean([-2,-1]).detach().cpu()
    return dic

def channel_activations(layers_2d,feature_extractor,imgs,batch_size=4):
    """
    Mean (over pixels) channel activations (per layer) of images provided

    Args:
        layers_2d: list of layers to extract
        feature_extractor: feature extractor
        imgs: images to pass through feature extractor
        batch_size: batch size

    Returns:
        outs_2d: dictionary with keys layers_2d where each value is of size (N_imgs,N_channels)
    """
    
    #some parameters
    device=list(feature_extractor.parameters())[0].device
    N=imgs.shape[0]
    
    #start a dictionary
    features=feature_extractor(imgs[:batch_size].to(device))
    outs_2d=dic_mean_2d(features,layers_2d) #mean activation of each channel in dictionary with layers as keys

    #go through remaining images and concatenate into outs_2d
    imgs=imgs[batch_size:] 
    print('Making predictions from stimuli to sort channels by standard deviation') 
    for i in tqdm(range(batch_size,N,batch_size)):
        features=feature_extractor(imgs[i:i+batch_size].to(device))
        for layer in layers_2d:
            outs_2d[layer]=torch.cat([outs_2d[layer],features[layer].mean([-2,-1]).detach().cpu()]) #concatenate 
    return outs_2d


def sorter(activations,max_channels=None,max_percent=None):
    """
    Sort channel activations by standard deviation

    Args:
        activations: dictionary of channel activations with keys as layers
        max_channels: maximum number of channels to take
        max_percent: maximum percentage of channels to take (priority over max_channels)

    Returns:
        sd: dictionary of sorted channel indices
    """
    sd = {}
    if max_percent is None:
        if max_channels is None:
            for key in activations.keys():
                sd[key] = activations[key].std(0).sort(descending=True)[1] #we are grabbing the inds
        else:
            for key in activations.keys():
                aa = activations[key].std(0).sort(descending=True)[1]
                sd[key] = aa[:max_channels]
    else:
        for key in activations.keys():
            aa=activations[key].std(0).sort(descending=True)[1]
            max_channels=int(len(aa)*max_percent/100)
            sd[key] = aa[:max_channels]
    return sd

def channels_to_use(layers_2d,feature_extractor,imgs,batch_size=1,max_channels=None,max_percent=None):
    """
    Returns a dictionary of sorted channel indices where each key is a provided layer to extract

    Args:
        layers_2d: list of layers to extract
        feature_extractor: feature extractor
        imgs: images to pass through feature extractor
        batch_size: batch size
        max_channels: maximum number of channels to take
        max_percent: maximum percentage of channels to take (priority over max_channels)
    """
    #TODO: maybe add cuda at this level if availble
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    activations_2d=channel_activations(layers_2d,feature_extractor.to(device),imgs,batch_size)
    sd_2d=sorter(activations_2d,max_channels,max_percent)
    return sd_2d


def map_times_rf(x,rf):
    """
    Function to multiply activation maps and rf-fields

    Args:
        x: map with size [B,C,H,W]
        rf: rf-field with size [Nv,H,W]

    Returns:
        <x|rf>: inner product with size [B,C,Nv]
    """
    
    #get parameters
    B,C,H,W = x.shape
    Nv=rf.shape[0]

    #reshape things
    x=x.view(B*C,H*W)
    rf=rf.view(Nv,H*W)

    #matrix multiplication
    out=matmul(x.float(),rf.T)

    return out.view(B,C,Nv)
    

def layer_shapes(feature_extractor,input_size):
    """
    Returns the shape of each feauture in a dictionary
    """
    shapes=OrderedDict()
    device=list(feature_extractor.parameters())[0].device
    _=feature_extractor(torch.rand(input_size,device=device))
    for key,value in _.items():
        shapes[key]=value.shape
    return shapes

def unique_2d_layer_shapes(layer_names,layer_shapes):
    """
    This function extracts information about unique 2d resolutions

    Args:
        layer_names: list of layer_names to extract
        layer_shapes: dictionary of layer shapes

    Returns:

        unique_sizes: list of unique resolutions
        layer_shapes: dictionary with the index of unique_sizes for each layer
        channels: total number of channels in each layer

    :return: list of unique resolutions and a dictionary with index of unique res of each layer
    """
    #get the unique 2d sizes
    unique_sizes=[]
    channels=OrderedDict()
    for key in layer_names:
        val_=layer_shapes[key]
        if val_.__len__()==4:
            val=val_[-2:]
            channels[key]=val_[1]
            if val not in unique_sizes:
                unique_sizes.append(tuple(val))
    unique_sizes=order_size_array(unique_sizes)
    #get the index dictionary
    inds_dic=OrderedDict()
    for key in layer_names:
        val_=layer_shapes[key]
        if val_.__len__()==4:
            val=tuple(val_[-2:])
            check=torch.asarray([val==a for a in unique_sizes])
            ind=torch.where(check==True)[0]
            inds_dic[key]=int(ind)
    return unique_sizes,inds_dic,channels

def order_size_array(sizes):
    sizes_sum=torch.asarray([torch.asarray(s).sum() for s in sizes])
    return [sizes[i] for i in sizes_sum.argsort(descending=True)]
