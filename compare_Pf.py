import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import itertools
import sys
import os
import errno
import h5py
import scipy.linalg as sl
import scipy.special as sp   
from f_enkf_isenRSW import gaspcohn_matrix, gaspcohn_sqrt_matrix
from isen_func import interp_sig2etab
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

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
n_ens = config.n_ens
n_obs = config.n_obs
n_obs_sat = config.n_obs_sat
k = config.k

###################################################################

n_job = int(sys.argv[2])-1
print(n_job)
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
print(indices)
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

###################################################################

# make fig directory (if it doesn't already exist)
dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))
figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

###################################################################

### load look-up table
h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

Xforec = np.load(str(dirn+'/X_forec.npy')) # long term forecast
X = Xforec[:,:,48,1]
Xbar = X.mean(axis=1)
Xdev = X - np.repeat(Xbar,n_ens).reshape(n_d,n_ens) # deviations
print(Xdev.shape)
print(Xbar.shape)

###################################################################

loc_rho = loc[i] # loc_rho is form of lengthscale.
rho = gaspcohn_matrix(loc_rho,Nk_fc,Neq)
print('loc matrix rho shape: ', np.shape(rho))

H = np.load(str(dirn+'/H_obs_oper_v2.npy'))

sig_mask = list(range(0,Nk_fc))
sigu_mask = list(range(Nk_fc,2*Nk_fc))
sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

K = n_ens

W = gaspcohn_sqrt_matrix(rho,n_d)
L = np.size(W,axis=1)
M = L*K
Z = np.zeros((n_d,M))
for j in range(L):
    Z[:,(j*K):((j+1)*K)] = (1./np.sqrt(K-1)) * W[:,j,None] * Xdev
v = np.array([Xbar + np.sqrt(M)*Z[:,k] for k in range(M)]).T
#    for k in range(M): plt.plot(range(800),v[:,k])
#    plt.show()
#    exit()
#    print(Z[sig_mask,:])
#    print(v[sig_mask,:])
#    print(v.shape)
vbar = np.repeat(np.mean(v,axis=1),M).reshape(n_d,M)
# print(vbar.shape)
vdev = v - vbar
 
### Manipulating v[sig_mask] to get radiance from each layer
veta2 = interp_sig2etab(v[sig_mask,:],h5_file_data)
veta1 = interp_sig2etab(v[sig_mask,:],h5_file_data) - v[sig_mask,:]
vB2 = veta2**k
vB1 = veta1**k

### Alpha weights
alpha1 = 0.5-0.5*sp.erf(-95*v[sig_mask,:]+21.5)
alpha2 = 0.425+0.425*sp.erf(-95*v[sig_mask,:]+21.5)
alpha3 = 0.5+0.5*sp.erf(-5*v[sig_mask,:]+3)
alpha4 = 0.5+0.5*sp.erf(-3*v[sig_mask,:]-1.16)

### Compute net radiance and replace in v[sig_mask]
vBsat = vB1*alpha3*alpha1 + vB2*(alpha4+alpha2)
v[sig_mask,:] = vB2

### Compute the projection of the model onto the observation space
Hv = np.zeros((n_obs,M)) 
if(n_obs_sat>0): 
    Hv[0:n_obs_sat,:] = np.matmul(H[0:n_obs_sat,sig_mask],vBsat)
Hv[n_obs_sat:,:] = np.matmul(H[n_obs_sat:,:],v)

Hvbar = np.repeat(np.mean(Hv,axis=1),M).reshape(n_obs, M)
Hvdev = Hv - Hvbar
Pf_loc = np.matmul(vdev,vdev.T)/M
Pf = rho*np.matmul(Xdev,Xdev.T)/(K-1)

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].text(0.03,0.75,'$\sigma$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].text(0.03,0.57,'$\sigma u$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].text(0.03,0.39,'$\sigma v$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].text(0.03,0.21,'$\sigma r$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].text(0.11,0.06,'$\sigma$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].text(0.19,0.06,'$\sigma u$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].text(0.29,0.06,'$\sigma v$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].text(0.39,0.06,'$\sigma r$',transform=plt.gcf().transFigure,fontsize=15)
axes[0].imshow(Pf_loc,cmap=plt.cm.RdBu,vmin=-1e-3,vmax=1e-3)
axes[0].set_title('Modulated ensemble',fontsize=18)
axes[0].set_yticks([0,200,400,600,800])
axes[0].set_yticklabels([])
axes[0].set_xticks([0,200,400,600,800])
axes[0].set_xticklabels([])
axes[1].text(0.54,0.06,'$\sigma$',transform=plt.gcf().transFigure,fontsize=15)
axes[1].text(0.63,0.06,'$\sigma u$',transform=plt.gcf().transFigure,fontsize=15)
axes[1].text(0.73,0.06,'$\sigma v$',transform=plt.gcf().transFigure,fontsize=15)
axes[1].text(0.82,0.06,'$\sigma r$',transform=plt.gcf().transFigure,fontsize=15)
im = axes[1].imshow(Pf,cmap=plt.cm.RdBu,vmin=-1e-3,vmax=1e-3)
axes[1].set_title('Original ensemble',fontsize=18)
axes[1].set_yticks([0,200,400,600,800])
axes[1].set_yticklabels([])
axes[1].set_xticks([0,200,400,600,800])
axes[1].set_xticklabels([])
plt.subplots_adjust(left=0.05,wspace=0.05)
cbar_ax = fig.add_axes([0.9, 0.11, 0.02, 0.77])
cbar = plt.colorbar(im,cax=cbar_ax,format='%.4f')
cbar.set_ticks([0.001,0.0008,0.0006,0.0004,0.0002,0.0,-0.0002,-0.0004,-0.0006,-0.0008,-0.001])
plt.show()
