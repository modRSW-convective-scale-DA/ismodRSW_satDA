#######################################################################
# Investigating effect of localisation lengthscale
#######################################################################

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import math as m
import numpy as np
import importlib.util
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import errno
import sys

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

loc = config.loc
add_inf = config.add_inf
rtpp = config.rtpp
rtps = config.rtps
IAU_dir = config.IAU_dir
outdir = config.outdir
Neq = config.Neq
n_d = config.n_d
Nk_fc = config.Nk_fc
n_ens = config.n_ens
n_obs = config.n_obs


### Derive position in parameter list (I,j,k,l) using the index passed via command line
n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
I = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

### 2. Choose time: plot at assimilation cycle ii
ii = int(sys.argv[3])

### Make fig directory (if it doesn't already exist)
dirn = '{}{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, IAU_dir, str(loc[I]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))# make fig directory (if it doesn't already exist)

figsdir = str(dirn+'/figs')

try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

##################################################################
# GASPARI-COHN TAPER FUNCTION FOR COV LOCALISATION
################################################################    
def gaspcohn(r):
    # Gaspari-Cohn taper function.
    # very close to exp(-(r/c)**2), where c = sqrt(0.15)
    # r should be >0 and normalized so taper = 0 at r = 1
    rr = 2.0*r
    rr += 1.e-13 # avoid divide by zero warnings from numpy
    taper = np.where(r<=0.5, \
                     ( ( ( -0.25*rr +0.5 )*rr +0.625 )*rr -5.0/3.0 )*rr**2 + 1.0,\
                     np.zeros(r.shape,r.dtype))

    taper = np.where(np.logical_and(r>0.5,r<1.), \
                    ( ( ( ( rr/12.0 -0.5 )*rr +0.625 )*rr +5.0/3.0 )*rr -5.0 )*rr \
                    + 4.0 - 2.0 / (3.0 * rr), taper)
    return taper    

##################################################################
# LOCALISATION PROPERTIES
################################################################

X = np.load(str(dirn+'/X_array.npy')) # fc ensembles
X_tr = np.load(str(dirn+'/X_tr_array.npy')) # truth
Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
Y_obs = np.load(str(outdir+IAU_dir+'/Y_obs_2xres_1h.npy')) # obs
OI = np.load(str(dirn+'/OI.npy')) # obs ensembles

print('X_array shape (n_d,n_ens,T)      : ', np.shape(X)) 
print('X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr)) 
print('Xan_array shape (n_d,n_ens,T)    : ', np.shape(Xan)) 
print('Y_obs_array shape (p,T)          : ', np.shape(Y_obs)) 
print('OI shape (Neq + 1,T)             : ', np.shape(OI)) 

##################################################################

Kk_fc = 1./Nk_fc 
n_ens = np.shape(X)[1]
n_obs = np.shape(Y_obs)[0]
t_an = np.shape(X)[2]
time_vec = list(range(0,t_an))

# masks for locating model variables in state vector
sig_mask = list(range(0,Nk_fc))
sigu_mask = list(range(Nk_fc,2*Nk_fc))
sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

##################################################################

# compute means and deviations
Xbar = np.empty(np.shape(X))
Xdev = np.empty(np.shape(X))
Pf = np.empty((n_d,n_d,n_ens))
Cf = np.empty((n_d,n_d,n_ens))

ONE = np.ones([n_ens,n_ens])
ONE = ONE/n_ens # NxN array with elements equal to 1/N
for T in time_vec[1:]:
    Xbar[:,:,T] = np.dot(X[:,:,T],ONE) # fc mean
    Xdev[:,:,T] = X[:,:,T] - Xbar[:,:,T] # fc deviations from mean

T = time_vec[int(sys.argv[3])]

### Calculation of  covariance matrix (with self-exclusion)
for i in range(0,n_ens):
    Pf[:,:,i] = np.matmul(np.delete(Xdev[:,:,T],i,1), np.delete(Xdev[:,:,T],i,1).T)
    Pf[:,:,i] = Pf[:,:,i] / (n_ens - 2)

Pf_ave = np.mean(Pf,axis=2)

print(Pf_ave.shape)

print('max Pf value: ', np.max(Pf_ave))
print('min Pf value: ', np.min(Pf_ave))

# correlation matrix
for i in range(0,n_ens):
    Cf[:,:,i] = np.corrcoef(np.delete(Xdev[:,:,T],i,1))

Cf_ave = np.mean(Cf,axis=2)

#taper functions
taper = np.zeros([Nk_fc,len(loc)])
for ii in range(0,len(loc)):
    loc_rho = loc[ii] # loc_rho is form of lengthscale.
    rr = np.arange(0,loc_rho,loc_rho/Nk_fc) 
    taper[:,ii] = gaspcohn(rr)

vec = taper[:,I]
rho = np.zeros((Nk_fc,Nk_fc))
for ii in range(Nk_fc):
    for jj in range(Nk_fc):
        rho[ii,jj] = vec[np.min([abs(ii-jj),abs(ii+Nk_fc-jj),abs(ii-Nk_fc-jj)])]
rho = np.tile(rho, (Neq,Neq))
print('loc matrix rho shape: ', np.shape(rho))

Pf_ave_loc = rho*Pf_ave
Cf_ave_loc = rho*Cf_ave


### The code below can plot either the taper functions or the covariance matrix
print(' *** PLOT: forecast correlation matrix ***')
tick_loc = [0,0.5*Nk_fc,1.5*Nk_fc,2.5*Nk_fc,3.5*Nk_fc,800]
tick_lab = ['',r'$\sigma$',r'$u$',r'$v$',r'$r$','']

fig, axes = plt.subplots(1, 2, figsize=(5,2.7))
fig.suptitle('$L_{loc}$= '+str(loc[I]))
fig.tight_layout(pad=0.8)

### PLOT TAPER FUNCTIONS
#for ii in range(0,len(loc)):
#    axes[0].plot(list(range(0,Nk_fc)), taper[:,ii], linewidth=1, label='$L_{loc}$ = %s' %loc[ii])
#axes[0].set_xticks([0,50,100,150,200])
#axes[0].set_xticklabels([0,50,100,150,200],fontsize=7)
#axes[0].set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
#axes[0].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=7)
#axes[0].set_xlabel('Distance (gridpoints)',fontsize=7)
#axes[0].set_ylabel('value',fontsize=7)
#axes[0].set_title('Taper functions',fontsize=8)
#axes[0].legend(loc = 1,fontsize=6)
#axes[0].set_aspect(1./axes[0].get_data_ratio())
#axes[0].tick_params(width=0.1)

#im=axes[1].imshow(rho,cmap=cm.Blues,vmin=0, vmax=1)
#axes[1].set_xticks(tick_loc)
#axes[1].set_yticks(tick_loc)
#axes[1].set_xticklabels(tick_lab,fontsize=7)
#axes[1].set_yticklabels(tick_lab,fontsize=7)
#plt.setp(fig.axes[1].get_xticklines(),visible=False) # invisible ticks
#plt.setp(fig.axes[1].get_yticklines(),visible=False)
#axes[1].set_title(r'$\rho$ with $L_{loc}$ = %s' %loc[I],fontsize=8)

### PLOT COVARIANCE MATRIX
invD=np.diag(1.0 / np.sqrt(np.diagonal(Pf_ave)))
im=axes[0].imshow(np.matmul(invD, np.matmul(Pf_ave, invD)), cmap=plt.cm.RdBu,
                    vmin=-1.0, vmax=1.0)
axes[0].set_xticks(tick_loc)
axes[0].set_yticks(tick_loc)
axes[0].set_xticklabels(tick_lab,fontsize=10)
axes[0].set_yticklabels(tick_lab,fontsize=10)
axes[0].hlines(199,0,799,color='black',linestyle='dashed')
axes[0].hlines(399,0,799,color='black',linestyle='dashed')
axes[0].hlines(599,0,799,color='black',linestyle='dashed')
axes[0].vlines(199,0,799,color='black',linestyle='dashed')
axes[0].vlines(399,0,799,color='black',linestyle='dashed')
axes[0].vlines(599,0,799,color='black',linestyle='dashed')
axes[0].set_title('Correlation matrix of $\mathbf{P}^f_e$',fontsize=10)

invD=np.diag(1.0 / np.sqrt(np.diagonal(Pf_ave_loc)))
im=axes[1].imshow(np.matmul(invD, np.matmul(Pf_ave_loc, invD)), cmap=plt.cm.RdBu,
                    vmin=-1.0, vmax=1.0)
axes[1].set_xticks(tick_loc)
axes[1].set_yticks(tick_loc)
axes[1].set_xticklabels(tick_lab,fontsize=10)
axes[1].set_yticklabels([])
axes[1].hlines(199,0,799,color='black',linestyle='dashed')
axes[1].hlines(399,0,799,color='black',linestyle='dashed')
axes[1].hlines(599,0,799,color='black',linestyle='dashed')
axes[1].vlines(199,0,799,color='black',linestyle='dashed')
axes[1].vlines(399,0,799,color='black',linestyle='dashed')
axes[1].vlines(599,0,799,color='black',linestyle='dashed')
axes[1].set_title('Correlation matrix of $\\rho \circ \mathbf{P}^f_e$',fontsize=10)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.2, 0.025, 0.6])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)

name_f = "/T%d_Pf.png" %T
f_name_f = str(figsdir+name_f)
plt.savefig(f_name_f,dpi=300)

print('Localisation matrix saved in: ',f_name_f)
