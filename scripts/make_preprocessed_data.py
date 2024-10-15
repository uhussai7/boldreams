
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nsdhandling import core
from nsdhandling.config import *

#get subject
subj=sys.argv[1]

#have to figure out how many uptos
subj_beta_path=os.path.join(BETA_ROOT,'subj%02d'%int(subj),FUNC_RES,FUNC_PREPROC)
sessions=os.listdir(subj_beta_path)
upto=max([int(s.split('betas_session')[1].split('.nii.gz')[0]) for s in sessions if 'betas_session' in s and 'hdf5' not in s])
print('Loading from %d sessions'%upto)

#load the stimulus
print('Loading raw stim data...')
stim_data=core.StimData([int(subj)])
print('Done.')

#load the fmri data
subj_list=[int(subj)]
print('Loading fmri data')
#stim_data.load_stimuli_raw(subj_list)
nsd_data=core.NsdData(subj_list,upto=upto)
nsd_data.load_from_raw(stim_data)
print('Done.')

#Save preprocessed
print('Saving...')
nsd_data.save_preprocessed()
print('Done')
