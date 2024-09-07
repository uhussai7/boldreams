import numpy as np
import nibabel as nib
from .file_utility import load_mask_from_nii,view_data

def inflate1d(x):
    y = x.astype(np.float32)
    return 1.*y + 0.5*np.roll(y,-1,axis=0) + 0.5*np.roll(y,1,axis=0)

def inflate2d(x):
    y = inflate1d(x)
    return 1.*y + 0.5*np.roll(y,-1,axis=1) + 0.5*np.roll(y,1,axis=1)

def inflate3d(x):
    y = inflate2d(x)
    return 1.*y + 0.5*np.roll(y,-1,axis=2) + 0.5*np.roll(y,1,axis=2)

def inflate_masks(s,mask_root):
    brain_mask_full   = load_mask_from_nii(mask_root + "/subj%02d/func1pt8mm/brainmask.nii.gz"%s)
    brain_seg_full    = load_mask_from_nii(mask_root + "/subj%02d/func1pt8mm/aseg.nii.gz"%s)
    brain_roi_full    = load_mask_from_nii(mask_root + "/subj%02d/func1pt8mm/roi/Kastner2015.nii.gz"%s)
    general_mask_full = load_mask_from_nii(mask_root + "/subj%02d/func1pt8mm/roi/nsdgeneral.nii.gz"%s)

    brain_nii_shape = brain_mask_full.shape
    brain_mask_full   = brain_mask_full.flatten().astype(bool)
    general_mask_full = (general_mask_full==1).flatten().astype(bool)

    voxel_seg_full = (brain_seg_full!=0).flatten()
    voxel_idx_full = np.arange(len(brain_mask_full))
    voxel_gen_full = (general_mask_full==1).flatten().astype(bool)
    voxel_roi_full = brain_roi_full.flatten()
    ###
    tight_mask = np.logical_or(np.logical_or(voxel_seg_full, voxel_gen_full), voxel_roi_full>=0)
    volume_tight_mask = view_data(brain_nii_shape, voxel_idx_full, tight_mask)
    volume_inflated_mask = inflate3d(volume_tight_mask)>=1.
    inflated_mask = volume_inflated_mask.flatten()
    ###
    print ("   tight  \t<  inflated  \t<  original")
    print ("   %d  \t<  %d  \t<  %d"%(np.sum(tight_mask), np.sum(volume_inflated_mask.flatten()), np.sum(brain_mask_full.flatten())))
    _ = view_data(brain_nii_shape, voxel_idx_full, inflated_mask, save_to=mask_root + "/subj%02d/func1pt8mm/brainmask_inflated_1.0.nii"%s)
