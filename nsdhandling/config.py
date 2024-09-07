import os



#Please define root paths
NSD_ROOT=os.environ.get('NSD_ROOT_PATH') #set NSD_ROOT_PATH as environment variable 
BETA_ROOT=os.path.join(NSD_ROOT,'nsddata_betas','ppdata')
MASK_ROOT=os.path.join(NSD_ROOT,'nsddata','ppdata')
STIM_ROOT=os.path.join(NSD_ROOT,'nsddata_stimuli','stimuli','nsd')
FREESURFER_ROOT=os.path.join(NSD_ROOT,'freesurfer') #freesurfer path for nsd subjects
#NSD_PREPROC=os.path.join(NSD_ROOT,'nsd_preproc')
SCRATCH_PATH=os.environ.get('SCRATCH_PATH') #using scratch for now
#NSD_PREPROC=os.path.join(SCRATCH_PATH,'nsd_preproc') #using scratch for now 
NSD_PREPROC=os.environ.get('NSD_PREPROC_PATH') #set NSD_ROOT_PATH as environment variable 


#Files
EXP_DESIGN_FILE=os.path.join(NSD_ROOT,'nsddata','experiments','nsd','nsd_expdesign.mat')
STIM_FILE=os.path.join(STIM_ROOT,'nsd_stimuli.hdf5')
STIM_INFO_FILE=os.path.join(NSD_ROOT,'nsddata','experiments','nsd','nsd_stim_info_merged.pkl')
COCO_TRN_ANN=os.path.join(NSD_ROOT,'coco_anns','captions_train2017.json')
COCO_VAL_ANN=os.path.join(NSD_ROOT,'coco_anns','captions_val2017.json')

#Fmri data parameters
FUNC_RES='func1pt8mm'
FUNC_PREPROC='betas_fithrf_GLMdenoise_RR'

#File extension for data
LOAD_EXT='.nii.gz'

#ROI handling
ROI_NAMES = ['V1', 'V2', 'V3', 'hV4', 'V3ab', 'LO', 'IPS', 'VO', 'PHC', 'MT', 'MST','other']
ROI_NAMEA_IDS = [[1, 2], [3, 4], [5, 6], [7], [16, 17], [14, 15], [18, 19, 20, 21, 22, 23], [8, 9], [10, 11], [13], [12], [24,25, 0]]
