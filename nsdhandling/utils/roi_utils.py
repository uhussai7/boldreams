import numpy as np
import torch
import nibabel as nib
from .nsd_orig.file_utility import load_mask_from_nii, view_data
from collections import OrderedDict
from ..config import *
import os

def read_ctab(path):
    """
    Reads a freesurfer ctab file

    Args:
        path: Path of ctab file

    Returns:
        indices: Indices corresponding to labels
        names: Label names
    """
    int_list=[]
    string_list=[]
    with open(path,'r') as file:
        for line in file:
            parts=line.strip().split(' ')
            if len(parts)==2:
                integer=int(parts[0])
                string=parts[1]
                if string != 'Unknown':
                    int_list.append(integer)
                    string_list.append(string)
    return int_list,string_list

def update_mask(mask,rois,subj):
    """
    Add rois to mask

    Args:
        mask: Mask to update (3D array)
        rois: List of rois
        subj" Subject id

    Returns:
        mask: Updated mask (3D array)
    """
    for roi in rois:
        path=os.path.join(MASK_ROOT,'subj%02d'%subj,FUNC_RES,'roi',roi+'.nii.gz')
        mask[load_mask_from_nii(path)>0]=1
    return mask

def update_roi_dic(mask,roi_dic,rois,subj):
    """
    Updates roi dictionary based on rois 

    Args:
        mask: Mask to update (3D array)
        roi_dic: Roi dictionary to update (contains flattened masks)
        rois: List of rois
        subj: Subject id

    Returns:
        roi_dic: Updated roi dictionary
    """
    mask=mask.flatten()
    for roi in rois:
        path=os.path.join(MASK_ROOT,'subj%02d'%subj,FUNC_RES,'roi',roi+'.nii.gz')
        roi_nii=load_mask_from_nii(path).flatten()
        if roi == 'basic_rois':
            inds,keys=read_ctab( FREESURFER_ROOT+ '/subj%02d/label/%s.mgz.ctab'%(subj,'Kastner2015'))
        else:
            inds, keys = read_ctab(FREESURFER_ROOT+ '/subj%02d/label/%s.mgz.ctab' % (subj, roi))
        for k,key in enumerate(keys):
            roi_dic[key]=roi_nii[mask==1]==inds[k]
    return roi_dic

def combine_basic_rois(roi_dic,roi_list,subj):
    """
    Updates roi dictionary to combine basic rois into coarser rois

    Args:
        roi_dic: Roi dictionary to update (contains flattened masks)
        rois: List of rois
        subj: Subject id

        Returns:
        roi_dic: Updated roi dictionary with basic rois combined
    """
    roi_dic_out={}
    basic_grouping={'V1':['V1v','V1d'],'V2':['V2v','V2d'],'V3':['V3v','V3d'],'V3ab':['V3A','V3B'],
                    'hV4':['hV4'], 'VO':['VO1','VO2'],
                    'LO': ['LO1', 'LO2'],
                    'PHC':['PHC1','PHC2'],
                    'IPS':['IPS0','IPS1','IPS2','IPS3','IPS4','IPS5'],
                    'MT':['TO1'],'MST':['TO2']}
    for roi in roi_list:
        if roi=='basic_rois':
            for key in basic_grouping:
                roi_dic_out[key]=combine_filters([roi_dic[k] for k in basic_grouping[key]])
        else:
            inds, keys = read_ctab(NSD_ROOT + '/freesurfer/subj%02d/label/%s.mgz.ctab' % (subj, roi))
            roi_dic_out[roi]=combine_filters([roi_dic[k] for k in keys])
    return roi_dic_out

def combine_filters(filters):
    #helper function to combine filters
    out=np.zeros_like(filters[0]).astype(bool)
    for f in filters:
        out[f]=1
    return out