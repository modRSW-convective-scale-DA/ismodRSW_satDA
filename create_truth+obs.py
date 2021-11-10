##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import sys
import importlib.util
import h5py
from scipy import signal

# HANDLE WARNINGS AS ERRORS
##################################################################
import warnings
warnings.filterwarnings("error")

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from f_isenRSW import make_grid
from f_enkf_isenRSW import generate_truth
from create_readme import create_readme
from obs_oper import obs_generator_rad

#################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE                     #
#################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
Nk_tr = config.Nk_tr
Nk_fc = config.Nk_fc
L = config.L
cfl_tr = config.cfl_tr
Neq = config.Neq
ic = config.ic
S0 = config.S0
A = config.A
V = config.V
sig_c = config.sig_c
sig_r = config.sig_r
cc2 = config.cc2
beta = config.beta
alpha2 = config.alpha2
g = config.g
R = config.R
theta1 = config.theta1
theta2 = config.theta2
eta0 = config.eta0
Z0 = config.Z0
U_scale = config.U_scale
k = config.k
Ro = config.Ro
tau_rel = config.tau_rel
tmax = config.tmax
assim_time = config.assim_time
dtmeasure = config.dtmeasure
ass_freq = config.ass_freq
Nmeas = config.Nmeas
Nforec = config.Nforec
n_obs = config.n_obs
dres = config.dres
n_d = config.n_d
ob_noise = config.ob_noise
sat_vel = config.sat_vel
n_obs_sat = config.n_obs_sat
n_obs_grnd = config.n_obs_grnd
n_d_grnd = config.n_d_grnd
n_obs_T = config.n_obs_T
n_obs_u = config.n_obs_u
n_obs_v = config.n_obs_v
n_obs_r = config.n_obs_r
obs_T_d = config.obs_T_d
obs_u_d = config.obs_u_d
obs_v_d = config.obs_v_d
obs_r_d = config.obs_r_d
sigr_obs_mask = config.sigr_obs_mask
sat_init_pos = config.sat_init_pos
swathe = config.swathe
U_relax = config.U_relax

#################################################################
# create directory for output
#################################################################
#check if dir exixts, if not make it
try:
    os.makedirs(outdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

##################################################################    
# Mesh generation and IC for truth 
##################################################################
tr_grid =  make_grid(Nk_tr,L) # truth
Kk_tr = tr_grid[0]
x_tr = tr_grid[1]
xc_tr = tr_grid[2]

### Truth ic
U0_tr, B_tr = ic(x_tr,Nk_tr,Neq,S0,L,A,V)
np.save(str(outdir+'/B_tr'),B_tr) #save topog for plotting

### Relaxation solution
U_rel = U_relax(Neq,Nk_tr,L,V,xc_tr,U0_tr)

U_tr_array = np.empty([Neq,Nk_tr,Nmeas+Nforec+1])
U_tr_array[:,:,0] = U0_tr

f_path_name = str(outdir+'/U_tr_array_2xres_'+ass_freq+'.npy')
f_obs_name = str(outdir+'/Y_obs_2xres_'+ass_freq+'.npy')

### LOAD LOOK-UP TABLE
h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

try:
    print(' *** Loading truth trajectory... *** ')
    U_tr_array = np.load(f_path_name)
except:
    print(' *** Generating truth trajectory... *** ')
    U_tr_array = generate_truth(U_tr_array, Nk_tr, tr_grid, Neq, cfl_tr, assim_time, tmax, dtmeasure, f_path_name, R, k, theta1, theta2, eta0, g, Z0, U_scale, sig_r, sig_c, cc2, alpha2, beta, Ro, U_rel, tau_rel, h5_file_data)

##################################################################    
# Pseudo-observations
##################################################################

print('Total no. of obs. =', n_obs)

# Sample the truth trajectory on the forecast grid.
U_tmp = np.copy(U_tr_array)

# for assimilation, work with [h,u,r]
U_tmp[1:,:,:] = U_tmp[1:,:,:]/U_tmp[0,:,:]

try:
    Y_obs = np.load(f_obs_name)
except:
    # create imperfect observations by adding the same observation noise
    # to each member's perfect observation. N.B. The observation time index
    # is shifted by one cycle relative to the truth trajectory because
    # no observations are assimilated at the initial condition time.
    Y_obs = np.empty([n_obs, np.size(U_tr_array, 2)-1])
    ob_noise = np.repeat(ob_noise,[n_obs_sat,n_obs_T,n_obs_u,n_obs_v,n_obs_r])
    for T in range(np.size(U_tr_array, 2)-1):
        Y_obs[:,T] = obs_generator_rad(U_tmp,Nk_tr,Nk_fc,Kk_tr,dres,Y_obs,ob_noise,sat_vel,n_obs_sat,swathe,n_obs_grnd,n_obs_r,n_d_grnd,obs_T_d,obs_u_d,obs_v_d,obs_r_d,sigr_obs_mask,T,k,sig_c,sig_r,sat_init_pos,h5_file_data)
    np.save(f_obs_name, Y_obs)
#### END OF THE PROGRAM ####