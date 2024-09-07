from yacs.config import CfgNode as CN
import os 

#Declare some nodes
_C = CN()
_C.SYSTEM = CN()
_C.PATHS = CN()
_C.FMRI=CN()
_C.TRAIN=CN()
_C.BACKBONE=CN()
_C.DREAMS=CN()

#Paths
_C.PATHS.NSD_ROOT=os.environ.get('NSD_ROOT_PATH') #set NSD_ROOT_PATH as environment variable 
_C.PATHS.BETA_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'nsddata_betas','ppdata')
_C.PATHS.MASK_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'nsddata','ppdata')
_C.PATHS.STIM_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'nsddata_stimuli','stimuli','nsd')
_C.PATHS.FREESURFER_ROOT=os.path.join(_C.PATHS.NSD_ROOT,'freesurfer') #freesurfer path for nsd subjects
_C.PATHS.BACKBONE_FILES=os.environ.get('BACKBONE_ROOT_PATH')
_C.PATHS.NSD_PREPROC=os.environ.get('NSD_PREPROC_PATH') #environment variable 
_C.PATHS.NSD_ENCODER=os.environ.get('NSD_ENCODER_PATH') #environment variable #this is where models are saved
_C.PATHS.NSD_CONFIG=os.environ.get('NSD_CONFIG_PATH')
_C.PATHS.NSD_TEXT=os.environ.get('NSD_TEXT_PATH') #this is to store text related stuff, text embeddings etc

#Files
_C.PATHS.EXP_DESIGN_FILE=os.path.join(_C.PATHS.NSD_ROOT,'nsddata','experiments','nsd','nsd_expdesign.mat')
_C.PATHS.STIM_FILE=os.path.join(_C.PATHS.STIM_ROOT,'nsd_stimuli.hdf5')

#Fmri data parameters
_C.FMRI.FUNC_RES='func1pt8mm'
_C.FMRI.FUNC_PREPROC='betas_fithrf_GLMdenoise_RR'

#File extension for data
_C.FMRI.LOAD_EXT='.nii.gz'

#Training params
_C.TRAIN.MAX_EPOCHS=10
_C.TRAIN.BATCH_SIZE=16

#Default backbone
_C.BACKBONE.NAME='alexnet'
_C.BACKBONE.FILE= 'alexnet.pt'   
_C.BACKBONE.FINETUNE= False
_C.BACKBONE.TEXT= False
_C.BACKBONE.LAYERS_TO_EXTRACT= ['features.2','features.5','features.7','features.9','features.12']
_C.BACKBONE.PERCENT_OF_CHANNELS= 100
_C.BACKBONE.INPUT_SIZE= (227,227)
_C.BACKBONE.NT=None #output vector size of text encoder

#Dreams
_C.DREAMS.ROTATE=1
_C.DREAMS.TRANSLATE=(0.1,0.1)
_C.DREAMS.SCALE=(1,1)

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

