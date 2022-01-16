##################################################################
### Plotting routines for saved data - Snapshots of forecast at given lead time
##################################################################
'''
Plotting routine: <plot_forec_x>

Loads saved data in specific directories and produces plots as a function of x at a given time.

NOTE: Any changes to the outer loop parameters should be replicated here too.

NOTE: currently saves as .png files

USE: python3 plot_forec_x.py <config_file> <index> <initialisation_time> <lead_time>

Assumes only one RTPP value.
'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import os
import errno
import h5py
import numpy as np
import matplotlib
import itertools
import importlib.util
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import scipy.special as sp

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################
from crps_calc_fun import crps_calc
from isen_func import interp_sig2etab 

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
Kk_fc = config.Kk_fc
Nmeas = config.Nmeas
Nforec = config.Nforec
assim_time = config.assim_time
n_d = config.n_d
ob_noise = config.ob_noise
sig_c = config.sig_c
sig_r = config.sig_r
n_ens = config.n_ens
n_obs = config.n_obs
n_obs_sat = config.n_obs_sat
n_obs_grnd = config.n_obs_grnd
n_obs_u = config.n_obs_u
n_obs_v = config.n_obs_v
n_obs_r = config.n_obs_r
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
table_file_name = config.table_file_name

### Derive position in parameter list (i,j,k,l) using the index passed via command line
n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

### Define initialisation time
ii = int(sys.argv[3])

### Define lead time: plot forecast at lead time jj
jj = int(sys.argv[4])

### Make fig directory (if it doesn't already exist)
dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))
figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

### Load look-up table
h5_file = h5py.File('inversion_tables/'+table_file_name,'r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

### Load data
print('*** Loading saved data... ')
B = np.load(str(dirn+'/B.npy')) #topography
Xforec = np.load(str(dirn+'/X_forec.npy')) #long-range forecast
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensemble
Y_obs = np.load(str(outdir+'/Y_obs_2xres_1h.npy')) # obs
OI = np.load(str(dirn+'/OI.npy')) # OI

### Print shape of data arrays to terminal (sanity check)
print(' Check array shapes...')
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)) 
print('Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)) 
print('Y_obs shape (p,T-1)    : ', np.shape(Y_obs)) 
print(' ')

##################################################################

### Determine parameters from loaded parameters
xc = np.linspace(Kk_fc/2,L-Kk_fc/2,Nk_fc) 
t_an = np.shape(Xan)[2]
time_vec = list(range(0,t_an))
print('time_vec = ', time_vec)
print(' ')
T = time_vec[ii]
lead_time = jj

print(' *** Plotting at time T level = ', T+lead_time)
print(' *** Assim. time: ', assim_time[T+lead_time])

### Masks for locating model variables in state vector
sig_mask = list(range(0,Nk_fc))
sigu_mask = list(range(Nk_fc,2*Nk_fc))
sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

### Masks for locating obs locations
row_vec_T = list(range(obs_T_d, Nk_fc+1, obs_T_d))
row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
row_vec_v = list(range(2*Nk_fc+obs_v_d, 3*Nk_fc+1, obs_v_d))
row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))
sat_pos = list(((sat_init_pos*Nk_fc+sat_vel*Nk_fc*(T+1))%Nk_fc).astype(int))
row_vec = np.array(sat_pos+row_vec_T+row_vec_u+row_vec_v+row_vec_r)

##################################################################

Xforbar = np.empty(np.shape(Xforec))

### CALCULATING ERRORS AT DIFFERENT LEAD TIMES ###
Xforbar[:,:,T,lead_time] = np.repeat(Xforec[:,:,T,lead_time].mean(axis=1), n_ens).reshape(n_d, n_ens)

##################################################################
frac = 0.15 # alpha value for translucent plotting
##################################################################
### 6 panel subplot for evolution of 3 vars: fc and an
##################################################################

### Manipulating X, Xan, Xbar, Xanbar, X_tr to get radiance from each layer
eta2_fc = interp_sig2etab(Xforec[sig_mask,:,T,lead_time],h5_file_data)
eta2_bar = interp_sig2etab(Xforbar[sig_mask,0,T,lead_time],h5_file_data)
eta2_tr = interp_sig2etab(X_tr[sig_mask,0,T+lead_time],h5_file_data)
eta1_fc = eta2_fc - Xforec[sig_mask,:,T,lead_time]
eta1_bar = eta2_bar - Xforbar[sig_mask,0,T,lead_time]
eta1_tr = eta2_tr - X_tr[sig_mask,0,T+lead_time]
B2_fc = eta2_fc**kgas
B2_bar = eta2_bar**kgas
B2_tr = eta2_tr**kgas
B1_fc = eta1_fc**kgas
B1_bar = eta1_bar**kgas
B1_tr = eta1_tr**kgas

### Alpha weights
alpha1_fc = 0.5-0.5*sp.erf(-95*Xforec[sig_mask,:,T,lead_time]+21.5)
alpha2_fc = 0.425+0.425*sp.erf(-95*Xforec[sig_mask,:,T,lead_time]+21.5)
alpha3_fc = 0.5+0.5*sp.erf(-5*Xforec[sig_mask,:,T,lead_time]+3)
alpha4_fc = 0.5+0.5*sp.erf(-3*Xforec[sig_mask,:,T,lead_time]-1.16)
alpha1_bar = 0.5-0.5*sp.erf(-95*Xforbar[sig_mask,0,T,lead_time]+21.5)
alpha2_bar = 0.425+0.425*sp.erf(-95*Xforbar[sig_mask,0,T,lead_time]+21.5)
alpha3_bar = 0.5+0.5*sp.erf(-5*Xforbar[sig_mask,0,T,lead_time]+3)
alpha4_bar = 0.5+0.5*sp.erf(-3*Xforbar[sig_mask,0,T,lead_time]-1.16)
alpha1_tr = 0.5-0.5*sp.erf(-95*X_tr[sig_mask,0,T+lead_time]+21.5)
alpha2_tr = 0.425+0.425*sp.erf(-95*X_tr[sig_mask,0,T+lead_time]+21.5)
alpha3_tr = 0.5+0.5*sp.erf(-5*X_tr[sig_mask,0,T+lead_time]+3)
alpha4_tr = 0.5+0.5*sp.erf(-3*X_tr[sig_mask,0,T+lead_time]-1.16)

### Compute net radiance
Bsat_fc = B1_fc*alpha3_fc*alpha1_fc + B2_fc*(alpha2_fc+alpha4_fc)
Bsat_bar = B1_bar*alpha1_bar*alpha3_bar + B2_bar*(alpha2_bar+alpha4_bar)
Bsat_tr = B1_tr*alpha1_tr*alpha3_tr + B2_tr*(alpha2_tr+alpha4_tr)

fig, axes = plt.subplots(Neq+1, 1, figsize=(8,10))

axes[0].plot(xc, Bsat_fc[:,1:], 'b',alpha=frac)
axes[0].plot(xc, Bsat_fc[:,0], 'b',alpha=frac,label="fc. ens.")
axes[0].plot(xc, Bsat_bar, 'r',label="Ens. mean")
axes[0].plot(xc, Bsat_tr, 'g',label="Truth")
axes[0].set_ylabel('$I(x)$',fontsize=18)

axes[1].plot(xc, Xforec[sig_mask,1:,T,lead_time], 'b',alpha=frac)
axes[1].plot(xc, Xforec[sig_mask,0,T,lead_time], 'b',alpha=frac,label="fc. ens.")
axes[1].plot(xc, Xforbar[sig_mask,0,T,lead_time], 'r',label="Ens. mean")
axes[1].plot(xc, X_tr[sig_mask,0,T+lead_time], 'g',label="Truth")
axes[1].plot(xc,sig_c*np.ones(len(xc)),'k:')
axes[1].plot(xc,sig_r*np.ones(len(xc)),'k:')
axes[1].set_ylim([0,0.1+np.max(X_tr[sig_mask,:,T])])
axes[1].set_ylabel('$\sigma(x)$',fontsize=18)
axes[0].legend(loc = 1)

axes[2].plot(xc, Xforec[sigu_mask,:,T,lead_time], 'b',alpha=frac)
axes[2].plot(xc, Xforbar[sigu_mask,0,T,lead_time], 'r')
axes[2].plot(xc, X_tr[sigu_mask,:,T+lead_time], 'g')
axes[2].set_ylabel('$u(x)$',fontsize=18)
if(n_obs_u>0):
    axes[2].plot(xc[row_vec[sigu_obs_mask]-Nk_fc-1], Y_obs[sigu_obs_mask,T+lead_time], 'go',linewidth=2.0)
    axes[2].errorbar(xc[row_vec[sigu_obs_mask]-Nk_fc-1], Y_obs[sigu_obs_mask,T+lead_time], ob_noise[2],
                   fmt='go',linewidth=2.0)

axes[3].plot(xc, Xforec[sigv_mask,:,T,lead_time], 'b',alpha=frac)
axes[3].plot(xc, Xforbar[sigv_mask,0,T,lead_time], 'r')
axes[3].plot(xc, X_tr[sigv_mask,:,T+lead_time], 'g')
axes[3].set_ylabel('$v(x)$',fontsize=18)
if(n_obs_v>0):
    axes[3].plot(xc[row_vec[sigv_obs_mask]-2*Nk_fc-1], Y_obs[sigv_obs_mask,T+lead_time], 'go',linewidth=2.0)
    axes[3].errorbar(xc[row_vec[sigv_obs_mask]-2*Nk_fc-1], Y_obs[sigv_obs_mask,T+lead_time], ob_noise[3],
                   fmt='go',linewidth=2.0)

axes[4].plot(xc, Xforec[sigr_mask,:,T,lead_time], 'b',alpha=frac)
axes[4].plot(xc, Xforbar[sigr_mask,0,T,lead_time], 'r')
axes[4].plot(xc, X_tr[sigr_mask,:,T+lead_time], 'g')
axes[4].plot(xc,np.zeros(len(xc)),'k')
axes[4].set_ylabel('$r(x)$',fontsize=18)
axes[4].set_ylim([-0.025,0.02+np.max(X_tr[sigr_mask,0,T+lead_time])])
axes[4].set_xlabel('$x$',fontsize=18)
if(n_obs_r>0):
    axes[4].plot(xc[row_vec[sigr_obs_mask]-3*Nk_fc-1], Y_obs[sigr_obs_mask,T+lead_time], 'go',linewidth=2.0)
    axes[4].errorbar(xc[row_vec[sigr_obs_mask]-3*Nk_fc-1], Y_obs[sigr_obs_mask,T+lead_time], ob_noise[4], fmt='go',
                  linewidth=2.0)

name_f = "/T"+str(T)+"_assim_lead+"+str(lead_time)+".png"
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f)
print(' ')
print(' *** %s at time level %d saved to %s' %(name_f,T,figsdir))

