'''
Main run script for batch-processing of EnKF jobs. 
Define outer-loop through parameter space and run the EnKF subroutine for each case.
Summary:
> truth generated outside of outer loop as this is the same for all experiments 
> uses subroutine <subr_enkf_modRSW_p> that parallelises ensemble forecasts using multiprocessing module
> Data saved to automatically-generated directories and subdirectories with accompanying readme.txt file.
'''

from __future__ import print_function

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
from builtins import str
from builtins import range
from scipy import signal
#from pynverse import inversefunc
import numpy as np
import os
import errno
import h5py
import importlib.util
import sys

##################################################################
# HANDLE WARNINGS AS ERRORS
##################################################################
import warnings
import gc
warnings.filterwarnings("error")

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from f_isenRSW import make_grid 
from f_enkf_isenRSW import generate_truth
from init_cond_isenRSW import init_cond_4
from create_readme import create_readme
from subr_enkf_isenRSW_p import run_enkf
from obs_oper import obs_generator_rad
from isen_func import interp_sig2etab, M_int

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
ic = config.ic
add_inf = config.add_inf
rtpp = config.rtpp
rtps = config.rtps
ass_freq = config.ass_freq

#################################################################
# create directory for output
#################################################################

# check if outdir exixts, if not make it
try:
    os.makedirs(outdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

### LOAD TRUTH, OBSERVATIONS AND OBSERVATION OPERATOR ###
f_path_name = str(outdir+'/U_tr_array_2xres_'+ass_freq+'.npy')
f_obs_name = str(outdir+'/Y_obs_2xres_'+ass_freq+'.npy')

### Check whether truth trajectory already exists, otherwise exit with error
try:
    ' *** Loading truth trajectory... *** '
    U_tr_array = np.load(f_path_name)
except:
    print(' Failed to find the truth trajectory: run create_truth+obs.py first')

### Check whether observations alrady exsist, otherwise exit with error
try:
    Y_obs = np.load(f_obs_name)
except:
    print('Failed to load the observations: run create_truth+obs.py first')

### Load look-up table for conversion from pseudo-density to non-dimensional pressure
h5_file = h5py.File('inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

##################################################################    
# EnKF: outer loop 
##################################################################
print(' ')
print(' ------- ENTERING EnKF OUTER LOOP ------- ')  
print(' ')
for i in range(0,len(loc)):
    for j in range(0,len(add_inf)):
        for m in range(0,len(rtpp)):
            for l in range(0,len(rtps)):
                run_enkf(i, j, m, l, U_tr_array, Y_obs, outdir, sys.argv[1], h5_file_data)
print(' ')   
print(' --------- FINISHED OUTER LOOP -------- ')
print(' ')   
print(' ------------ END OF PROGRAM ---------- ')  
print(' ') 
    
##################################################################    
#                        END OF PROGRAM                          #
##################################################################
