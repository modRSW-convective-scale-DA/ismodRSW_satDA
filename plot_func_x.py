##################################################################
#--------------- Plotting routines for saved data ---------------
##################################################################
'''
Plotting routine: <plot_func_x>

Loads saved data in specific directories and produces plots as a function of x at a given assimilation time. To use, specify (1) dir_name, (2) combination of parameters ijk, (3) time level T = time_vec[ii], i.e., choose ii.

NOTE: Any changes to the outer loop parameters should be replicated here too.

NOTE: currently saves as .png files

CALL WITH: 'python plot_func_x <i> <j> <k>'

Assumes only one RTPP value.
'''

# generic modules 
import matplotlib
matplotlib.use('Agg')
import h5py
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

L = config.L
outdir = config.outdir
loc = config.loc
add_inf = config.add_inf
rtpp = config.rtpp
rtps = config.rtps
Neq = config.Neq
dres = config.dres
Nk_fc = config.Nk_fc
Kk_fc = config.	Kk_fc
Nmeas = config.Nmeas
Nforec = config.Nforec
assim_time = config.assim_time
n_d = config.n_d
ob_noise = config.ob_noise
sig_c = config.sig_c
sig_r = config.sig_r
n_ens = config.n_ens
n_obs = config.n_obs
n_obs_T = config.n_obs_T
n_obs_sat = config.n_obs_sat
n_obs_u = config.n_obs_u
n_obs_v = config.n_obs_v
n_obs_r = config.n_obs_r
n_obs_grnd = config.n_obs_grnd
obs_dens = config.o_d
obs_T_d = config.obs_T_d
obs_u_d = config.obs_u_d
obs_v_d = config.obs_v_d
obs_r_d = config.obs_r_d
sat_init_pos = config.sat_init_pos
sat_vel = config.sat_vel
sig_obs_mask = config.sat_obs_mask
sigT_obs_mask = config.sigT_obs_mask
sigu_obs_mask = config.sigu_obs_mask
sigv_obs_mask = config.sigv_obs_mask
sigr_obs_mask = config.sigr_obs_mask
kgas = config.k
ass_freq = config.ass_freq

## 1. CHOOSE ijkl. E.g., for test_enkf1111/ [i,j,k,l] = [0,0,0,0]
n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

## 2. CHOOSE time: plot at assimilation cycle ii
ii = int(sys.argv[3])
##################################################################

dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]), str(add_inf[j]), str(rtpp[k]), str(rtps[l]))
figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

### load look-up table
h5_file = h5py.File('inversion_tables/sigma_eta_theta2_291_theta1_311_eta0_0.48_Z0_6120_k_0.29.hdf','r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

## load data
print('*** Loading saved data... ')
B = np.load(str(dirn+'/B.npy')) #topography
X = np.load(str(dirn+'/X_array.npy')) # fc ensemble
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
Y_obs = np.load(str(outdir+'/Y_obs_2xres_'+ass_freq+'.npy')) # obs ensembles
OI = np.load(str(dirn+'/OI.npy')) # OI

# print shape of data arrays to terminal (sanity check)
print(' Check array shapes...')
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('X_array shape (n_d,n_ens,T)      : ', np.shape(X)) 
print('X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)) 
print('Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)) 
print('Y_obs shape (p,T-1)    : ', np.shape(Y_obs)) 
print(' ')
##################################################################

# determine parameters from loaded parameters
xc = np.linspace(Kk_fc/2,L-Kk_fc/2,Nk_fc)
t_an = np.shape(X)[2]
time_vec = list(range(0,t_an))
print('time_vec = ', time_vec)
print(' ')
T = time_vec[ii]

print(' *** Plotting at time T level = ', T)
print(' *** Assim. time: ', assim_time[T])

# masks for locating model variables in state vector
sig_mask = list(range(0,Nk_fc))
sigu_mask = list(range(Nk_fc,2*Nk_fc))
sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

# masks for locating obs locations
row_vec_T = list(range(obs_T_d, Nk_fc+1, obs_T_d))
row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
row_vec_v = list(range(2*Nk_fc+obs_v_d, 3*Nk_fc+1, obs_v_d))
row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))
sat_pos = list(((sat_init_pos*Nk_fc+sat_vel*Nk_fc*(T+1))%Nk_fc).astype(int))
row_vec = np.array(sat_pos+row_vec_T+row_vec_u+row_vec_v+row_vec_r)
##################################################################

# compute means and deviations
Xbar = np.empty(np.shape(X))
Xdev = np.empty(np.shape(X))
Xanbar = np.empty(np.shape(X))
Xandev = np.empty(np.shape(X))
Xdev_tr = np.empty(np.shape(X))
Xandev_tr = np.empty(np.shape(X))

#ONE = np.ones([n_ens,n_ens])
#ONE = ONE/n_ens # NxN array with elements equal to 1/N
for ii in time_vec:
#    Xbar[:,:,ii] = np.dot(X[:,:,ii],ONE) # fc mean
    Xbar[:,:,ii] = np.repeat(X[:,:,ii].mean(axis=1), n_ens).reshape(n_d, n_ens)
    Xdev[:,:,ii] = X[:,:,ii] - Xbar[:,:,ii] # fc deviations from mean
    Xdev_tr[:,:,ii] = X[:,:,ii] - X_tr[:,:,ii] # fc deviations from truth
#    Xanbar[:,:,ii] = np.dot(Xan[:,:,ii],ONE) # an mean
    Xanbar[:,:,ii] = np.repeat(Xan[:,:,ii].mean(axis=1), n_ens).reshape(n_d, n_ens)
    Xandev[:,:,ii] = Xan[:,:,ii] - Xanbar[:,:,ii] # an deviations from mean
    Xandev_tr[:,:,ii] = Xan[:,:,ii] - X_tr[:,:,ii] # an deviations from truth

##################################################################
frac = 0.15 # alpha value for translucent plotting
##################################################################
### 6 panel subplot for evolution of 3 vars: fc and an
##################################################################

### Manipulating X, Xan, Xbar, Xanbar, X_tr to get radiance from each layer
eta2_fc = interp_sig2etab(X[sig_mask,:,T],h5_file_data)
eta2_bar = interp_sig2etab(Xbar[sig_mask,0,T],h5_file_data)
eta2_tr = interp_sig2etab(X_tr[sig_mask,0,T],h5_file_data)
eta2_an = interp_sig2etab(Xan[sig_mask,:,T],h5_file_data)
eta2_anbar = interp_sig2etab(Xanbar[sig_mask,0,T],h5_file_data)
eta1_fc = eta2_fc - X[sig_mask,:,T]
eta1_bar = eta2_bar - Xbar[sig_mask,0,T]
eta1_tr = eta2_tr - X_tr[sig_mask,0,T]
eta1_an = eta2_an - Xan[sig_mask,:,T]
eta1_anbar = eta2_anbar - Xanbar[sig_mask,0,T]
B2_fc = eta2_fc**kgas
B2_bar = eta2_bar**kgas
B2_tr = eta2_tr**kgas
B2_an = eta2_an**kgas
B2_anbar = eta2_anbar**kgas
B1_fc = eta1_fc**kgas
B1_bar = eta1_bar**kgas
B1_tr = eta1_tr**kgas
B1_an = eta1_an**kgas
B1_anbar = eta1_anbar**kgas

### Alpha weights
#alpha1_fc = 0.5+0.1*np.exp(-70*((X[sig_mask,:,T]-sig_r)**2)/(sig_r-sig_c))
#erf_fc = 0.5+0.5*sp.erf(-10*X[sig_mask,:,T]+3.8)
#alpha2_fc = erf_fc*alpha1_fc
#alpha1_bar = 0.5+0.1*np.exp(-70*((Xbar[sig_mask,0,T]-sig_r)**2)/(sig_r-sig_c))
#erf_bar = 0.5+0.5*sp.erf(-10*Xbar[sig_mask,0,T]+3.8)
#alpha2_bar = erf_bar*alpha1_bar
#alpha1_tr = 0.5+0.1*np.exp(-70*((X_tr[sig_mask,0,T]-sig_r)**2)/(sig_r-sig_c))
#erf_tr = 0.5+0.5*sp.erf(-10*X_tr[sig_mask,0,T]+3.8)
#alpha2_tr = erf_tr*alpha1_tr
#alpha1_an = 0.5+0.1*np.exp(-70*((Xan[sig_mask,:,T]-sig_r)**2)/(sig_r-sig_c))
#erf_an = 0.5+0.5*sp.erf(-10*Xan[sig_mask,:,T]+3.8)
#alpha2_an = erf_an*alpha1_an
#alpha1_anbar = 0.5+0.1*np.exp(-70*((Xanbar[sig_mask,0,T]-sig_r)**2)/(sig_r-sig_c))
#erf_anbar = 0.5+0.5*sp.erf(-10*Xanbar[sig_mask,0,T]+3.8)
#alpha2_anbar = erf_anbar*alpha1_anbar
alpha1_fc = 0.5-0.5*sp.erf(-95*X[sig_mask,:,T]+21.5)
alpha2_fc = 0.425+0.425*sp.erf(-95*X[sig_mask,:,T]+21.5)
alpha3_fc = 0.5+0.5*sp.erf(-5*X[sig_mask,:,T]+3)
alpha4_fc = 0.5+0.5*sp.erf(-3*X[sig_mask,:,T]-1.16)
alpha1_bar = 0.5-0.5*sp.erf(-95*Xbar[sig_mask,0,T]+21.5)
alpha2_bar = 0.425+0.425*sp.erf(-95*Xbar[sig_mask,0,T]+21.5)
alpha3_bar = 0.5+0.5*sp.erf(-5*Xbar[sig_mask,0,T]+3)
alpha4_bar = 0.5+0.5*sp.erf(-3*Xbar[sig_mask,0,T]-1.16)
alpha1_tr = 0.5-0.5*sp.erf(-95*X_tr[sig_mask,0,T]+21.5)
alpha2_tr = 0.425+0.425*sp.erf(-95*X_tr[sig_mask,0,T]+21.5)
alpha3_tr = 0.5+0.5*sp.erf(-5*X_tr[sig_mask,0,T]+3)
alpha4_tr = 0.5+0.5*sp.erf(-3*X_tr[sig_mask,0,T]-1.16)
alpha1_an = 0.5-0.5*sp.erf(-95*Xan[sig_mask,:,T]+21.5)
alpha2_an = 0.425+0.425*sp.erf(-95*Xan[sig_mask,:,T]+21.5)
alpha3_an = 0.5+0.5*sp.erf(-5*Xan[sig_mask,:,T]+3)
alpha4_an = 0.5+0.5*sp.erf(-3*Xan[sig_mask,:,T]-1.16)
alpha1_anbar = 0.5-0.5*sp.erf(-95*Xanbar[sig_mask,0,T]+21.5)
alpha2_anbar = 0.425+0.425*sp.erf(-95*Xanbar[sig_mask,0,T]+21.5)
alpha3_anbar = 0.5+0.5*sp.erf(-5*Xanbar[sig_mask,0,T]+3)
alpha4_anbar = 0.5+0.5*sp.erf(-3*Xanbar[sig_mask,0,T]-1.16) 

### Compute net radiance
#Bsat_fc = B1_fc*alpha1_fc + B2_fc*alpha2_fc
#Bsat_bar = B1_bar*alpha1_bar + B2_bar*alpha2_bar
#Bsat_tr = B1_tr*alpha1_tr + B2_tr*alpha2_tr
#Bsat_an = B1_an*alpha1_an + B2_an*alpha2_an
#Bsat_anbar = B1_anbar*alpha1_anbar + B2_anbar*alpha2_anbar
Bsat_fc = B1_fc*alpha3_fc*alpha1_fc + B2_fc*(alpha2_fc+alpha4_fc)
Bsat_bar = B1_bar*alpha1_bar*alpha3_bar + B2_bar*(alpha2_bar+alpha4_bar)
Bsat_tr = B1_tr*alpha1_tr*alpha3_tr + B2_tr*(alpha2_tr+alpha4_tr)
Bsat_an = B1_an*alpha1_an*alpha3_an + B2_an*(alpha2_an+alpha4_an)
Bsat_anbar = B1_anbar*alpha1_anbar*alpha3_anbar + B2_anbar*(alpha2_anbar+alpha4_anbar)

print(B1_tr)
print(B2_tr)
print(Bsat_tr)
print(sat_pos)
print(Y_obs[sig_obs_mask,T])
print(Y_obs[sig_obs_mask,T-1])
print(Y_obs[sig_obs_mask,T+1])

fig, axes = plt.subplots(Neq+1, 2, figsize=(15,10), gridspec_kw = {'height_ratios':[1, 3, 3, 3, 3]})
#plt.suptitle("Ensemble trajectories (t = %s, N = %s): [od, loc, inf] = [%s, %s, %s]" % (assim_time[T],n_ens,o_d[i], loc[j], inf[k]),fontsize=16)

axes[0,0].plot(xc, Bsat_fc[:,1:], 'b',alpha=frac)
axes[0,0].plot(xc, Bsat_fc[:,0], 'b',alpha=frac,label="fc. ens.")
axes[0,0].plot(xc, Bsat_bar, 'r',label="Ens. mean")
axes[0,0].plot(xc, Bsat_tr, 'g',label="Truth")
if(n_obs_sat>0): axes[0,0].errorbar(xc[sat_pos], Y_obs[sig_obs_mask,T], ob_noise[0], fmt='go',linewidth=2.0,label="Obs.")
#axes[0,0].set_ylim([np.amin(np.concatenate((interp_sig2etab(X_tr[sig_mask,0,T]),Y_obs[sig_obs_mask,T])))-0.001,np.amax(np.concatenate((interp_sig2etab(X_tr[sig_mask,0,T]),Y_obs[sig_obs_mask,T])))+0.001])
axes[0,0].set_ylabel('$I_{sat}(x)$',fontsize=18)

axes[0,1].plot(xc, Bsat_an[:,1:], 'b',alpha=frac)
axes[0,1].plot(xc, Bsat_an[:,0], 'b',alpha=frac,label="an. ens.")
axes[0,1].plot(xc, Bsat_anbar, 'c',label="Analysis")
axes[0,1].plot(xc, Bsat_tr, 'g',label="Truth")
if(n_obs_sat>0): axes[0,1].errorbar(xc[sat_pos], Y_obs[sig_obs_mask,T], ob_noise[0], fmt='go',linewidth=2.0,label="Obs.")
#axes[0,1].set_ylim([1.031,1.034])
#axes[0,1].set_ylim([np.amin(np.concatenate((interp_sig2etab(X_tr[sig_mask,0,T]),Y_obs[sig_obs_mask,T])))-0.001,np.amax(np.concatenate((interp_sig2etab(X_tr[sig_mask,0,T]),Y_obs[sig_obs_mask,T])))+0.001])

#axes[1,0].plot(xc, B2_fc[:,1:], 'b',alpha=frac)
#axes[1,0].plot(xc, B2_fc[:,0], 'b',alpha=frac,label="fc. ens.")
#axes[1,0].plot(xc, B2_bar, 'r',label="Ens. mean")
#axes[1,0].plot(xc, B2_tr, 'g',label="Truth")
#if(n_obs_T>0): axes[1,0].errorbar(xc[row_vec[sigT_obs_mask]-1], Y_obs[sigT_obs_mask,T], ob_noise[1], fmt='go',linewidth=2.0,label="Obs.")
#axes[1,0].set_ylabel('$T_2(x)$',fontsize=18)

#axes[1,1].plot(xc, B2_an[:,1:], 'b',alpha=frac)
#axes[1,1].plot(xc, B2_an[:,0], 'b',alpha=frac,label="an. ens.")
#axes[1,1].plot(xc, B2_anbar, 'c',label="Analysis")
#axes[1,1].plot(xc, B2_tr, 'g',label="Truth")
#if(n_obs_T>0): axes[1,1].errorbar(xc[row_vec[sigT_obs_mask]-1], Y_obs[sigT_obs_mask,T], ob_noise[1], fmt='go',linewidth=2.0,label="Obs.")

axes[1,0].plot(xc, X[sig_mask,1:,T], 'b',alpha=frac)
axes[1,0].plot(xc, X[sig_mask,0,T], 'b',alpha=frac,label="fc. ens.")
axes[1,0].plot(xc, Xbar[sig_mask,0,T], 'r',label="Ens. mean")
axes[1,0].plot(xc, X_tr[sig_mask,0,T], 'g',label="Truth")
axes[1,0].plot(xc,sig_c*np.ones(len(xc)),'k:')
axes[1,0].plot(xc,sig_r*np.ones(len(xc)),'k:')
axes[1,0].set_ylim([0,0.1+np.max(X_tr[sig_mask,:,T])])
axes[1,0].set_ylabel('$\sigma(x)$',fontsize=18)

axes[1,1].plot(xc, Xan[sig_mask,1:,T], 'b',alpha=frac)
axes[1,1].plot(xc, Xan[sig_mask,0,T], 'b',alpha=frac,label="an. ens.")
axes[1,1].plot(xc, Xanbar[sig_mask,0,T], 'c',linewidth=2.0,label="Analysis")
axes[1,1].plot(xc, X_tr[sig_mask,0,T], 'g',label="Truth")
axes[1,1].plot(xc,sig_c*np.ones(len(xc)),'k:')
axes[1,1].plot(xc,sig_r*np.ones(len(xc)),'k:')
axes[1,1].set_ylim([0,0.1+np.max(X_tr[sig_mask,:,T])])

axes[2,0].plot(xc, X[sigu_mask,:,T], 'b',alpha=frac)
axes[2,0].plot(xc, Xbar[sigu_mask,0,T], 'r')
axes[2,0].plot(xc, X_tr[sigu_mask,:,T], 'g')
axes[2,0].set_ylabel('$u(x)$',fontsize=18)
if(n_obs_u>0):
    axes[2,0].plot(xc[row_vec[sigu_obs_mask]-Nk_fc-1], Y_obs[sigu_obs_mask,T], 'go',linewidth=2.0)
    axes[2,0].errorbar(xc[row_vec[sigu_obs_mask]-Nk_fc-1], Y_obs[sigu_obs_mask,T], ob_noise[2], fmt='go',linewidth=2.0)

axes[2,1].plot(xc, Xan[sigu_mask,:,T], 'b',alpha=frac)
axes[2,1].plot(xc, Xanbar[sigu_mask,0,T], 'c',linewidth=2.0)
axes[2,1].plot(xc, X_tr[sigu_mask,:,T], 'g')
if(n_obs_u>0): axes[2,1].errorbar(xc[row_vec[sigu_obs_mask]-Nk_fc-1], Y_obs[sigu_obs_mask,T], ob_noise[2], fmt='go',linewidth=2.0)

axes[3,0].plot(xc, X[sigv_mask,:,T], 'b',alpha=frac)
axes[3,0].plot(xc, Xbar[sigv_mask,0,T], 'r')
axes[3,0].plot(xc, X_tr[sigv_mask,:,T], 'g')
axes[3,0].set_ylabel('$v(x)$',fontsize=18)
if(n_obs_v>0):    
    axes[3,0].plot(xc[row_vec[sigv_obs_mask]-2*Nk_fc-1], Y_obs[sigv_obs_mask,T], 'go',linewidth=2.0)
    axes[3,0].errorbar(xc[row_vec[sigv_obs_mask]-2*Nk_fc-1], Y_obs[sigv_obs_mask,T], ob_noise[3], fmt='go',linewidth=2.0)

axes[3,1].plot(xc, Xan[sigv_mask,:,T], 'b',alpha=frac)
axes[3,1].plot(xc, Xanbar[sigv_mask,0,T], 'c',linewidth=2.0)
axes[3,1].plot(xc, X_tr[sigv_mask,:,T], 'g')
if(n_obs_v>0): axes[3,1].errorbar(xc[row_vec[sigv_obs_mask]-2*Nk_fc-1], Y_obs[sigv_obs_mask,T], ob_noise[3], fmt='go',linewidth=2.0)

axes[4,0].plot(xc, X[sigr_mask,:,T], 'b',alpha=frac)
axes[4,0].plot(xc, Xbar[sigr_mask,0,T], 'r')
axes[4,0].plot(xc, X_tr[sigr_mask,:,T], 'g')
axes[4,0].plot(xc,np.zeros(len(xc)),'k')
axes[4,0].set_ylabel('$r(x)$',fontsize=18)
axes[4,0].set_ylim([-0.025,0.02+np.max(X_tr[sigr_mask,0,T])])
axes[4,0].set_xlabel('$x$',fontsize=18)
if(n_obs_r>0):
    axes[4,0].plot(xc[row_vec[sigr_obs_mask]-3*Nk_fc-1], Y_obs[sigr_obs_mask,T], 'go',linewidth=2.0)
    axes[4,0].errorbar(xc[row_vec[sigr_obs_mask]-3*Nk_fc-1], Y_obs[sigr_obs_mask,T], ob_noise[4],fmt='go',linewidth=2.0)

axes[4,1].plot(xc, Xan[sigr_mask,:,T], 'b',alpha=frac)
axes[4,1].plot(xc, Xanbar[sigr_mask,0,T], 'c',linewidth=2.0)
axes[4,1].plot(xc, X_tr[sigr_mask,:,T], 'g')
axes[4,1].plot(xc,np.zeros(len(xc)),'k')
axes[4,1].set_ylim([-0.025,0.02+np.max(X_tr[sigr_mask,0,T])])
axes[4,1].set_xlabel('$x$',fontsize=18)
if(n_obs_r>0):
    axes[4,1].errorbar(xc[row_vec[sigr_obs_mask]-3*Nk_fc-1], Y_obs[sigr_obs_mask,T], ob_noise[4],fmt='go',linewidth=2.0)

name_f = "/T%d_assim.png" %T
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f)
print(' ')
print(' *** %s at time level %d saved to %s' %(name_f,T,figsdir))

##################################################################
###                       ERRORS                              ####
##################################################################

## ANALYSIS
an_err = Xanbar[:,0,T] - X_tr[:,0,T] # an_err = analysis ens. mean - truth
an_err2 = an_err**2
# domain-averaged mean errors
an_ME_sig = an_err[sig_mask].mean() 
an_ME_sigu = an_err[sigu_mask].mean()
an_ME_sigv = an_err[sigv_mask].mean()
an_ME_sigr = an_err[sigr_mask].mean()
# domain-averaged absolute errors
an_absME_sig = np.absolute(an_err[sig_mask])
an_absME_sigu = np.absolute(an_err[sigu_mask])
an_absME_sigv = np.absolute(an_err[sigv_mask])
an_absME_sigr = np.absolute(an_err[sigr_mask])

# cov matrix
Pa = np.dot(Xandev[:,:,T],np.transpose(Xandev[:,:,T]))
Pa = Pa/(n_ens - 1) # analysis covariance matrix
var_an = np.diag(Pa)

Pa_tr = np.dot(Xandev_tr[:,:,T],np.transpose(Xandev_tr[:,:,T]))
Pa_tr = Pa_tr/(n_ens - 1) # fc covariance matrix w.r.t truth
var_ant = np.diag(Pa_tr)

## FORECAST
fc_err = Xbar[:,0,T] - X_tr[:,0,T] # fc_err = ens. mean - truth
fc_err2 = fc_err**2
# domain-averaged mean errors
fc_ME_sig = fc_err[sig_mask].mean()
fc_ME_sigu = fc_err[sigu_mask].mean()
fc_ME_sigv = fc_err[sigv_mask].mean()
fc_ME_sigr = fc_err[sigr_mask].mean()
# domain-averaged absolute errors
fc_absME_sig = np.absolute(fc_err[sig_mask])
fc_absME_sigu = np.absolute(fc_err[sigu_mask])
fc_absME_sigv = np.absolute(fc_err[sigv_mask])
fc_absME_sigr = np.absolute(fc_err[sigr_mask])

# cov matrix
Pf = np.dot(Xdev[:,:,T],np.transpose(Xdev[:,:,T]))
Pf = Pf/(n_ens - 1) # fc covariance matrix
var_fc = np.diag(Pf)

Pf_tr = np.dot(Xdev_tr[:,:,T],np.transpose(Xdev_tr[:,:,T]))
Pf_tr = Pf_tr/(n_ens - 1) # fc covariance matrix w.r.t. truth
var_fct = np.diag(Pf_tr)
'''
# fc/an
ME_ratio_h = np.sqrt(fc_err2[sig_mask])/np.sqrt(an_err2[h_mask])
ME_ratio_hu = np.sqrt(fc_err2[sigu_mask])/np.sqrt(an_err2[hu_mask])
ME_ratio_hr = np.sqrt(fc_err2[sigr_mask])/np.sqrt(an_err2[hr_mask])
# fc - an
ME_diff_h = np.sqrt(fc_err2[sig_mask])-np.sqrt(an_err2[h_mask])
ME_diff_hu = np.sqrt(fc_err2[sigu_mask])-np.sqrt(an_err2[hu_mask])
ME_diff_hr = np.sqrt(fc_err2[sigr_mask])-np.sqrt(an_err2[hr_mask])
'''
##################################################################

# fontsize
ft = 16

# position text on plot
pl_sig = np.max([np.sqrt(var_fc[sig_mask]),fc_absME_sig])
pl_sigu = np.max([np.sqrt(var_fc[sigu_mask]),fc_absME_sigu])
pl_sigv = np.max([np.sqrt(var_fc[sigv_mask]),fc_absME_sigv])
pl_sigr = np.max([np.sqrt(var_fc[sigr_mask]),fc_absME_sigr])

# domain-averaged errors
an_spr_sig = np.mean(np.sqrt(var_an[sig_mask]))
an_rmse_sig = np.mean(np.sqrt(var_ant[sig_mask]))
fc_spr_sig = np.mean(np.sqrt(var_fc[sig_mask]))
fc_rmse_sig = np.mean(np.sqrt(var_fct[sig_mask]))

an_spr_sigu = np.mean(np.sqrt(var_an[sigu_mask]))
an_rmse_sigu = np.mean(np.sqrt(var_ant[sigu_mask]))
fc_spr_sigu = np.mean(np.sqrt(var_fc[sigu_mask]))
fc_rmse_sigu = np.mean(np.sqrt(var_fct[sigu_mask]))

an_spr_sigv = np.mean(np.sqrt(var_an[sigv_mask]))
an_rmse_sigv = np.mean(np.sqrt(var_ant[sigv_mask]))
fc_spr_sigv = np.mean(np.sqrt(var_fc[sigv_mask]))
fc_rmse_sigv = np.mean(np.sqrt(var_fct[sigv_mask]))

an_spr_sigr = np.mean(np.sqrt(var_an[sigr_mask]))
an_rmse_sigr = np.mean(np.sqrt(var_ant[sigr_mask]))
fc_spr_sigr = np.mean(np.sqrt(var_fc[sigr_mask]))
fc_rmse_sigr = np.mean(np.sqrt(var_fct[sigr_mask]))

##################################################################
### 6 panel subplot: comparing spread and error for fc and an
##################################################################

fig, axes = plt.subplots(Neq, 2, figsize=(12,12))

axes[0,0].plot(xc, np.sqrt(var_fc[sig_mask]),'r',label='fc spread') # spread
axes[0,0].plot(xc, fc_absME_sig,'r--',label='fc err') # rmse
axes[0,0].plot(xc, np.sqrt(var_an[sig_mask]),'b',label='an spread')
axes[0,0].plot(xc, an_absME_sig,'b--',label='an err')
axes[0,0].set_ylabel('$\sigma(x)$',fontsize=18)
axes[0,0].text(0.025, 1.2*pl_sig, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_sig,np.mean(an_absME_sig)), fontsize=ft, color='b')
axes[0,0].text(0.025, 1.1*pl_sig, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_sig,np.mean(fc_absME_sig)), fontsize=ft, color='r')
axes[0,0].set_ylim([0,1.3*pl_sig])

axes[1,0].plot(xc, np.sqrt(var_fc[sigu_mask]), 'r')
axes[1,0].plot(xc, fc_absME_sigu, 'r--')
axes[1,0].plot(xc, np.sqrt(var_an[sigu_mask]), 'b')
axes[1,0].plot(xc, an_absME_sigu , 'b--')
axes[1,0].set_ylabel('$u(x)$',fontsize=18)
axes[1,0].text(0.025, 1.2*pl_sigu, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_sigu,an_absME_sigu.mean()), fontsize=ft, color='b')
axes[1,0].text(0.025, 1.1*pl_sigu, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_sigu,fc_absME_sigu.mean()), fontsize=ft, color='r')
axes[1,0].set_ylim([0,1.3*pl_sigu])

axes[2,0].plot(xc, np.sqrt(var_fc[sigv_mask]), 'r')
axes[2,0].plot(xc, fc_absME_sigv, 'r--')
axes[2,0].plot(xc, np.sqrt(var_an[sigv_mask]), 'b')
axes[2,0].plot(xc, an_absME_sigv , 'b--')
axes[2,0].set_ylabel('$u(x)$',fontsize=18)
axes[2,0].text(0.025, 1.2*pl_sigu, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_sigv,an_absME_sigv.mean()), fontsize=ft, color='b')
axes[2,0].text(0.025, 1.1*pl_sigv, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_sigv,fc_absME_sigv.mean()), fontsize=ft, color='r')
axes[2,0].set_ylim([0,1.3*pl_sigv])

axes[3,0].plot(xc, np.sqrt(var_fc[sigr_mask]), 'r')
axes[3,0].plot(xc, fc_absME_sigr , 'r--')
axes[3,0].plot(xc, np.sqrt(var_an[sigr_mask]), 'b')
axes[3,0].plot(xc, an_absME_sigr , 'b--')
axes[3,0].set_ylabel('$r(x)$',fontsize=18)
axes[3,0].set_xlabel('$x$',fontsize=18)
axes[3,0].text(0.025, 1.2*pl_sigr, '$(SPR,ERR)_{an} = (%.3g,%.3g)$' %(an_spr_sigr,an_absME_sigr.mean() ), fontsize=ft, color='b')
axes[3,0].text(0.025, 1.1*pl_sigr, '$(SPR,ERR)_{fc} = (%.3g,%.3g)$' %(fc_spr_sigr,fc_absME_sigr.mean() ), fontsize=ft, color='r')
axes[3,0].set_ylim([0,1.3*pl_sigr])

axes[0,1].plot(xc, fc_absME_sig - np.sqrt(var_fc[sig_mask]), 'r',label='fc: err  - spr')
axes[0,1].plot(xc, an_absME_sig - np.sqrt(var_an[sig_mask]), 'b',label='an: err - spr')
axes[0,1].plot(xc,np.zeros(len(xc)),'k:')
axes[0,1].legend(loc=0)

axes[1,1].plot(xc, fc_absME_sigu - np.sqrt(var_fc[sigu_mask]), 'r')
axes[1,1].plot(xc, an_absME_sigu - np.sqrt(var_an[sigu_mask]), 'b')
axes[1,1].plot(xc, np.zeros(len(xc)),'k:')

axes[2,1].plot(xc, fc_absME_sigv - np.sqrt(var_fc[sigv_mask]), 'r')
axes[2,1].plot(xc, an_absME_sigv - np.sqrt(var_an[sigv_mask]), 'b')
axes[2,1].plot(xc, np.zeros(len(xc)),'k:')

axes[3,1].plot(xc, fc_absME_sigr - np.sqrt(var_fc[sigr_mask]), 'r')
axes[3,1].plot(xc, an_absME_sigr - np.sqrt(var_an[sigr_mask]), 'b')
axes[3,1].plot(xc, np.zeros(len(xc)),'k:')
axes[3,1].set_xlabel('$x$',fontsize=18)

name_f = "/T%d_spr_err.png" %T
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f)
print(' ')
print(' *** %s at time level %d saved to %s' %(name_f,T,figsdir))


##################################################################
### 4 panel subplot: CRPS of 4 vars for fc and an
##################################################################

CRPS_fc = np.empty((Neq,Nk_fc))
CRPS_an = np.empty((Neq,Nk_fc))

for ii in sig_mask:
    CRPS_fc[0,ii] = crps_calc(X[ii,:,T],X_tr[ii,0,T])
    CRPS_fc[1,ii] = crps_calc(X[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
    CRPS_fc[2,ii] = crps_calc(X[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
    CRPS_fc[3,ii] = crps_calc(X[ii+3*Nk_fc,:,T],X_tr[ii+3*Nk_fc,0,T])
    CRPS_an[0,ii] = crps_calc(Xan[ii,:,T],X_tr[ii,0,T])
    CRPS_an[1,ii] = crps_calc(Xan[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
    CRPS_an[2,ii] = crps_calc(Xan[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
    CRPS_an[3,ii] = crps_calc(Xan[ii+3*Nk_fc,:,T],X_tr[ii+3*Nk_fc,0,T])

lw = 1. # linewidth
axlim0 = np.max(CRPS_fc[0,:])
axlim1 = np.max(CRPS_fc[1,:])
axlim2 = np.max(CRPS_fc[2,:])
axlim3 = np.max(CRPS_fc[3,:])
ft = 16
xl = 0.65

fig, axes = plt.subplots(4, 1, figsize=(7,12))

axes[0].plot(xc, CRPS_fc[0,:],'r',linewidth=lw,label='fc')
axes[0].plot(xc, CRPS_an[0,:],'b',linewidth=lw,label='an')
axes[0].set_ylabel('$h(x)$',fontsize=18)
axes[0].text(xl, 1.2*axlim0, '$CRPS_{an} = %.3g$' %CRPS_an[0,:].mean(axis=-1), fontsize=ft, color='b')
axes[0].text(xl, 1.1*axlim0, '$CRPS_{fc} = %.3g$' %CRPS_fc[0,:].mean(axis=-1), fontsize=ft, color='r')
axes[0].set_ylim([0,1.3*axlim0])

axes[1].plot(xc, CRPS_fc[1,:],'r',linewidth=lw)
axes[1].plot(xc, CRPS_an[1,:],'b',linewidth=lw)
axes[1].set_ylabel('$u(x)$',fontsize=18)
axes[1].text(xl, 1.2*axlim1, '$CRPS_{an} = %.3g$' %CRPS_an[1,:].mean(axis=-1), fontsize=ft, color='b')
axes[1].text(xl, 1.1*axlim1, '$CRPS_{fc} = %.3g$' %CRPS_fc[1,:].mean(axis=-1), fontsize=ft, color='r')
axes[1].set_ylim([0,1.3*axlim1])

axes[1].plot(xc, CRPS_fc[2,:],'r',linewidth=lw)
axes[1].plot(xc, CRPS_an[2,:],'b',linewidth=lw)
axes[1].set_ylabel('$v(x)$',fontsize=18)
axes[1].text(xl, 1.2*axlim2, '$CRPS_{an} = %.3g$' %CRPS_an[2,:].mean(axis=-1), fontsize=ft, color='b')
axes[1].text(xl, 1.1*axlim2, '$CRPS_{fc} = %.3g$' %CRPS_fc[2,:].mean(axis=-1), fontsize=ft, color='r')
axes[1].set_ylim([0,1.3*axlim2])

axes[3].plot(xc, CRPS_fc[3,:],'r',linewidth=lw)
axes[3].plot(xc, CRPS_an[3,:],'b',linewidth=lw)
axes[3].set_ylabel('$r(x)$',fontsize=18)
axes[3].text(xl, 1.2*axlim3, '$CRPS_{an} = %.3g$' %CRPS_an[3,:].mean(axis=-1), fontsize=ft, color='b')
axes[3].text(xl, 1.1*axlim3, '$CRPS_{fc} = %.3g$' %CRPS_fc[3,:].mean(axis=-1), fontsize=ft, color='r')
axes[3].set_ylim([0,1.3*axlim3])
axes[3].set_xlabel('$x$',fontsize=18)

name = "/T%d_crps.png" %T
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s at time level %d saved to %s' %(name,T,figsdir))
