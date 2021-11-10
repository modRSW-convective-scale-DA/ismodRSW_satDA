### This script computes the Degrees of Freedom for Signal (cf. eq.(26) in Migliorini, 2013)

# IMPORT GENERIC MODULES 
import matplotlib
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
import scipy.special as sp   
import itertools

from crps_calc_fun import crps_calc
from isen_func import *

##################################################################

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
loc = config.loc
add_inf = config.add_inf
rtpp = config.rtpp
rtps = config.rtps
Neq = config.Neq
Nk_fc = config.Nk_fc
n_d = config.n_d
ob_noise = config.ob_noise
n_ens = config.n_ens
n_obs = config.n_obs
n_obs_T = config.n_obs_T
n_obs_sat = config.n_obs_sat
n_obs_u = config.n_obs_u
n_obs_v = config.n_obs_v
n_obs_r = config.n_obs_r
obs_T_d = config.obs_T_d
obs_u_d = config.obs_u_d
obs_v_d = config.obs_v_d
obs_r_d = config.obs_r_d
sat_init_pos = config.sat_init_pos
sat_vel = config.sat_vel
sat_obs_mask = config.sat_obs_mask
sigT_obs_mask = config.sigT_obs_mask
sigu_obs_mask = config.sigu_obs_mask
sigv_obs_mask = config.sigv_obs_mask
sigr_obs_mask = config.sigr_obs_mask
kgas = config.k

## 1. CHOOSE ijkl. E.g., for test_enkf1111/ [i,j,k,l] = [0,0,0,0]
n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

##################################################################

dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]), str(add_inf[j]), str(rtpp[k]), str(rtps[l]))
figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

## load data
print('*** Loading saved data... ')
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
Xforec = np.load(str(dirn+'/X_forec.npy')) # long term forecast
OI = np.load(str(dirn+'/OI.npy')) # OI

# print shape of data arrays to terminal (sanity check)
print(' Check array shapes...')
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan))
print('X_forec array shape (n_d,n_ens,T,N_forec): ', np.shape(Xforec))
print(' ')
##################################################################

t_an = np.shape(Xan)[2]
time_vec = list(range(0,t_an))
print('time_vec = ', time_vec)
print(' ') 

### LOAD LOOK-UP TABLE
h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

# masks for locating model variables in state vector
sig_mask = list(range(0,Nk_fc))
sigu_mask = list(range(Nk_fc,2*Nk_fc))
sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

### Variable definition
Xforbar = np.empty(np.shape(Xforec))
HXfordev = np.empty((n_obs,n_ens,len(time_vec)))
S_sat = np.empty((n_obs_sat,n_ens,len(time_vec)))
S_u = np.empty((n_obs_u,n_ens,len(time_vec)))
S_v = np.empty((n_obs_v,n_ens,len(time_vec)))
S_r = np.empty((n_obs_r,n_ens,len(time_vec)))
S = np.empty((n_obs,n_ens,len(time_vec)))
ds_sat = np.empty(len(time_vec))
ds_u = np.empty(len(time_vec))
ds_v = np.empty(len(time_vec))
ds_r = np.empty(len(time_vec))
### CALCULATING ERRORS AT DIFFERENT LEAD TIMES ###
Xforbar = np.repeat(Xforec.mean(axis=1), n_ens).reshape(Xforec.shape)

### Compute root square of R    
ob_noise = np.repeat(ob_noise,[n_obs_sat,n_obs_T,n_obs_u,n_obs_v,n_obs_r])
A = (1/ob_noise)*np.identity(n_obs) # obs cov matrix

### Manipulating X, Xan, Xbar, Xanbar to get radiance from each layer
eta2_fc = interp_sig2etab(Xforec[sig_mask,:,:,:],h5_file_data)
eta2_bar = interp_sig2etab(Xforbar[sig_mask,0,:,:],h5_file_data)
eta1_fc = interp_sig2etab(Xforec[sig_mask,:,:,:],h5_file_data) - Xforec[sig_mask,:,:,:]
eta1_bar = interp_sig2etab(Xforbar[sig_mask,0,:,:],h5_file_data) - Xforbar[sig_mask,0,:,:]
B2_fc = eta2_fc**kgas
B2_bar = eta2_bar**kgas
B1_fc = eta1_fc**kgas
B1_bar = eta1_bar**kgas

### Alpha weights
alpha1_fc = 0.5-0.5*sp.erf(-95*Xforec[sig_mask,:,:,:]+21.5)
alpha2_fc = 0.425+0.425*sp.erf(-95*Xforec[sig_mask,:,:,:]+21.5) 
alpha3_fc = 0.5+0.5*sp.erf(-5*Xforec[sig_mask,:,:,:]+3)
alpha4_fc = 0.5+0.5*sp.erf(-3*Xforec[sig_mask,:,:,:]-1.16)
alpha1_bar = 0.5-0.5*sp.erf(-95*Xforbar[sig_mask,0,:,:]+21.5)
alpha2_bar = 0.425+0.425*sp.erf(-95*Xforbar[sig_mask,0,:,:]+21.5)
alpha3_bar = 0.5+0.5*sp.erf(-5*Xforbar[sig_mask,0,:,:]+3)
alpha4_bar = 0.5+0.5*sp.erf(-3*Xforbar[sig_mask,0,:,:]-1.16)

### Compute net radiance
Bsat_fc = B1_fc*alpha3_fc*alpha1_fc + B2_fc*(alpha2_fc+alpha4_fc)
Bsat_bar = B1_bar*alpha1_bar*alpha3_bar + B2_bar*(alpha2_bar+alpha4_bar)

### Replace pseudo-density with radiance
Xforec[sig_mask,:,:,:] = Bsat_fc
Xforbar[sig_mask,:,:,:] = np.repeat(Bsat_bar,n_ens).reshape(Xforbar[sig_mask,:,:,:].shape)

### Masks for locating obs locations
row_vec_T = list(range(obs_T_d, Nk_fc+1, obs_T_d))
row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
row_vec_v = list(range(2*Nk_fc+obs_v_d, 3*Nk_fc+1, obs_v_d))
row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))

for T in time_vec: 
    
    sat_pos = list(((sat_init_pos*Nk_fc+sat_vel*Nk_fc*(T+1))%Nk_fc).astype(int))
    row_vec = np.array(sat_pos+row_vec_T+row_vec_u+row_vec_v+row_vec_r)-1
    HXfordev[:,:,T] = Xforec[row_vec,:,T,1] - Xforbar[row_vec,:,T,1]
    if(n_obs_sat>0): S_sat[:,:,T] = np.matmul(A[sat_obs_mask,sat_obs_mask],HXfordev[sat_obs_mask,:,T])
    S_u[:,:,T] = np.matmul(A[sigu_obs_mask,sigu_obs_mask],HXfordev[sigu_obs_mask,:,T])
    S_v[:,:,T] = np.matmul(A[sigv_obs_mask,sigv_obs_mask],HXfordev[sigv_obs_mask,:,T])
    S_r[:,:,T] = np.matmul(A[sigr_obs_mask,sigr_obs_mask],HXfordev[sigr_obs_mask,:,T])
    S[:,:,T] = np.matmul(A,HXfordev[:,:,T])
    if(n_obs_sat>0): Esat,Gsat,VTsat = np.linalg.svd(S_sat[:,:,T])
    Eu,Gu,VTu = np.linalg.svd(S_u[:,:,T])
    Ev,Gv,VTv = np.linalg.svd(S_v[:,:,T])
    Er,Gr,VTr = np.linalg.svd(S_r[:,:,T])
    E,G,VT = np.linalg.svd(S[:,:,T])
    if(n_obs_sat>0): ds_sat[T] = sum(Gsat**2/(1+Gsat**2))
    ds_u[T] = sum(Gu**2/(1+Gu**2))
    ds_v[T] = sum(Gv**2/(1+Gv**2))
    ds_r[T] = sum(Gr**2/(1+Gr**2))
    ds = sum(G**2/(1+G**2))

plt.plot(time_vec,ds_sat,linestyle='solid',color='blue')
plt.plot(time_vec,ds_u,linestyle='solid',color='red')
plt.plot(time_vec,ds_v,linestyle='solid',color='green')
plt.plot(time_vec,ds_r,linestyle='solid',color='orange')
plt.show()
