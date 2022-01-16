##################################################################
### This script computes the error growth of ensemble foercasts
##################################################################
'''
    <EFS_stats>
    Computes and plots error growth for the EFS data produced in <run_ismodRSW_EFS>.
    Also computes and saves error doubling times, to be used in <err_doub_hist>.
    Needs <config_file> and <analysis_step> (selected initial condition) as arguments.
        '''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import os
import sys
import errno
import importlib.util
import itertools
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

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
n_d = config.n_d
Nk_fc = config.Nk_fc
n_ens = config.n_ens
Neq = config.Neq
Nforec = config.Nforec
Nmeas = config.Nmeas
dres = config.dres

### Derive position in parameter list (i,j,k,l) using the index passed via command line
n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

### Make fig directory (if it doesn't already exist)
dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))
dirEDT = str(dirn+'/EDT')
figsdir = str(dirEDT+'/figs')

### Check if dir exixts, if not make it
try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

T0 = int(sys.argv[3]) # Analysis step selected as initial condition
Tfc = 36 # Maximum lead time
EFS_data = '/X_EFS_array_T'+str(T0)+'.npy'

# LOAD DATA FROM GIVEN DIRECTORY
X = np.load(str(dirEDT+EFS_data)) # fc ensembles

##################################################################

Kk_fc = 1./Nk_fc
time_vec = list(range(0, Tfc))

# LOAD TRUTH TRAJECTORY
U_tr = np.load(str(outdir+'/U_tr_array_2xres_1h.npy'))

# Manipulate U_tr to get X_tr
U_tr_tmp = np.empty((Neq,Nk_fc,Nforec+Nmeas))
for jj in range(0,Nk_fc):
    U_tr_tmp[:,jj,:] = U_tr[:,jj*dres,1:]
U_tr_tmp[1:,:,:] = U_tr_tmp[1:,:,:]/U_tr_tmp[0,:,:]

X_tr_array = np.empty((n_d,1,Nmeas+Nforec))
for kk in range(0,Nmeas+Nforec):
    X_tr_array[:,0,kk] = U_tr_tmp[:,:,kk].flatten()

X_tr = X_tr_array
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('X shape (n_d,n_ens,T,Nforec)      : ', np.shape(X))
print('X_tr_array shape (n_d,1,T)       : ', np.shape(X_tr))

# masks for locating model variables in state vector
sig_mask = list(range(0,Nk_fc))
sigu_mask = list(range(Nk_fc,2*Nk_fc))
sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

##################################################################

# for means and deviations
Xforbar = np.empty(np.shape(X))
Xfordev = np.empty(np.shape(X))
Xfordev_tr = np.empty(np.shape(X))

# for errs as at each assim time
rmse_fc = np.empty((Neq,len(time_vec)))
spr_fc = np.empty((Neq,len(time_vec)))
ame_fc = np.empty((Neq,len(time_vec)))
tote_fc = np.empty((Neq,len(time_vec)))
error_fc = np.empty((Neq,n_ens,len(time_vec)))

ONE = np.ones([n_ens,n_ens])
ONE = ONE/n_ens # NxN array with elements equal to 1/N

print(' *** Calculating errors...')

for T in time_vec:
    
    plt.clf() # clear figs from previous loop
    
    Xforbar[:,:,T] = np.repeat(np.mean(X[:,:,T],axis=1),n_ens).reshape(n_d,n_ens) # fc mean
    Xfordev[:,:,T] = X[:,:,T] - Xforbar[:,:,T] # fc deviations from mean
    Xfordev_tr[:,:,T] = X[:,:,T] - X_tr[:,:,T0+T] # fc deviations from truth
    
    ##################################################################
    ###               ERRORS + SPREAD                             ####
    ##################################################################
    
    
    # FORECAST: ensemble mean error
    fc_err = Xforbar[:,0,T] - X_tr[:,0,T0+T] # fc_err = ens. mean - truth
    fc_err2 = fc_err**2

    # forecast cov matrix
    Pf = np.dot(Xfordev[:,:,T],np.transpose(Xfordev[:,:,T]))
    Pf = Pf/(n_ens - 1) # fc covariance matrix
    var_fc = np.diag(Pf)
    
    Pf_tr = np.dot(Xfordev_tr[:,:,T],np.transpose(Xfordev_tr[:,:,T]))
    Pf_tr = Pf_tr/(n_ens - 1) # fc covariance matrix w.r.t. truth
    var_fct = np.diag(Pf_tr)
    
    err2 = Xfordev_tr[:,:,T]**2
    err2_sig = err2[sig_mask,:].mean(axis=0)
    err2_u = err2[sigu_mask,:].mean(axis=0)
    err2_v = err2[sigv_mask,:].mean(axis=0)
    err2_r = err2[sigr_mask,:].mean(axis=0)
    
    error_fc[0,:,T] = np.sqrt(err2_sig)
    error_fc[1,:,T] = np.sqrt(err2_u)
    error_fc[2,:,T] = np.sqrt(err2_v)
    error_fc[3,:,T] = np.sqrt(err2_r)
    
    # domain-averaged stats
    ame_fc[0,T] = np.mean(np.absolute(fc_err[sig_mask]))
    spr_fc[0,T] = np.sqrt(np.mean(var_fc[sig_mask]))
    rmse_fc[0,T] = np.sqrt(np.mean(fc_err2[sig_mask]))
    tote_fc[0,T] = np.sqrt(np.mean(var_fct[sig_mask]))

    ame_fc[1,T] = np.mean(np.absolute(fc_err[sigu_mask]))
    spr_fc[1,T] = np.sqrt(np.mean(var_fc[sigu_mask]))
    rmse_fc[1,T] = np.sqrt(np.mean(fc_err2[sigu_mask]))
    tote_fc[1,T] = np.sqrt(np.mean(var_fct[sigu_mask]))

    ame_fc[2,T] = np.mean(np.absolute(fc_err[sigv_mask]))
    spr_fc[2,T] = np.sqrt(np.mean(var_fc[sigv_mask]))
    rmse_fc[2,T] = np.sqrt(np.mean(fc_err2[sigv_mask]))
    tote_fc[2,T] = np.sqrt(np.mean(var_fct[sigv_mask]))

    ame_fc[3,T] = np.mean(np.absolute(fc_err[sigr_mask]))
    spr_fc[3,T] = np.sqrt(np.mean(var_fc[sigr_mask]))
    rmse_fc[3,T] = np.sqrt(np.mean(fc_err2[sigr_mask]))
    tote_fc[3,T] = np.sqrt(np.mean(var_fct[sigr_mask]))

###########################################################################

print(' ')
print(' PLOT : RMS ERRORS (for each ensemble...)')
print(' ')
frac = 0.15
axlim0 = np.max(np.max(error_fc[0,:,:]))
axlim1 = np.max(np.max(error_fc[1,:,:]))
axlim2 = np.max(np.max(error_fc[2,:,:]))
axlim3 = np.max(np.max(error_fc[3,:,:]))

fig, axes = plt.subplots(Neq, 2, figsize=(12,12))
for ii in range(0,n_ens):
    axes[0,0].plot(time_vec, error_fc[0,ii,:],'r',alpha=frac)
axes[0,0].plot(time_vec, error_fc[0,:,:].mean(axis=0),'k',linewidth=2.)
axes[0,0].set_ylabel('$\sigma$',fontsize=18)
axes[0,0].set_title("Forecast Error, E(T)")

for ii in range(0,n_ens):
    axes[1,0].plot(time_vec, error_fc[1,ii,:],'r',alpha=frac)
axes[1,0].plot(time_vec, error_fc[1,:,:].mean(axis=0),'k',linewidth=2.)
axes[1,0].set_ylabel('$u$',fontsize=18)
axes[1,0].set_ylim([0,1.3*axlim1])

for ii in range(0,n_ens):
    axes[2,0].plot(time_vec, error_fc[2,ii,:],'r',alpha=frac) #
axes[2,0].plot(time_vec, error_fc[2,:,:].mean(axis=0),'k',linewidth=2.)
axes[2,0].set_ylabel('$v$',fontsize=18)
axes[2,0].set_xlabel('Assim. time $T$',fontsize=14)
axes[2,0].set_ylim([0,1.3*axlim2])

for ii in range(0,n_ens):
    axes[3,0].plot(time_vec, error_fc[3,ii,:],'r',alpha=frac) #
axes[3,0].plot(time_vec, error_fc[3,:,:].mean(axis=0),'k',linewidth=2.)
axes[3,0].set_ylabel('$r$',fontsize=18)
axes[3,0].set_xlabel('Assim. time $T$',fontsize=14)
axes[3,0].set_ylim([0,1.3*axlim3])

for ii in range(0,n_ens):
    axes[0,1].plot(time_vec, error_fc[0,ii,:]/error_fc[0,ii,0],'r',alpha=frac)
ed = error_fc[0,:,:]/error_fc[0,:,0].reshape(n_ens,1)
print(np.shape(ed))
axes[0,1].plot(time_vec, ed[:,:].mean(axis=0),'k',linewidth=2.)
axes[0,1].plot(time_vec,2*np.ones(len(time_vec)),'k--',linewidth=2.)
axes[0,1].set_ylabel('$h$',fontsize=18)
axes[0,1].set_title("Relative Forecast Error, E(T)/E(0)")

for ii in range(0,n_ens):
    axes[1,1].plot(time_vec, error_fc[1,ii,:]/error_fc[1,ii,0],'r',alpha=frac)
ed = error_fc[1,:,:]/error_fc[1,:,0].reshape(n_ens,1)
print(np.shape(ed))
axes[1,1].plot(time_vec, ed[:,:].mean(axis=0),'k',linewidth=2.)
axes[1,1].plot(time_vec,2*np.ones(len(time_vec)),'k--',linewidth=2.)
axes[1,1].set_ylabel('$u$',fontsize=18)

for ii in range(0,n_ens):
    axes[2,1].plot(time_vec, error_fc[2,ii,:]/error_fc[2,ii,0],'r',alpha=frac) #
ed = error_fc[2,:,:]/error_fc[2,:,0].reshape(n_ens,1)
print(np.shape(ed))
axes[2,1].plot(time_vec, ed[:,:].mean(axis=0),'k',linewidth=2.)
axes[2,1].plot(time_vec,2*np.ones(len(time_vec)),'k--',linewidth=2.)
axes[2,1].set_ylabel('$v$',fontsize=18)
axes[2,1].set_xlabel('Assim. time $T$',fontsize=14)

for ii in range(0,n_ens):
    axes[3,1].plot(time_vec, error_fc[3,ii,:]/error_fc[3,ii,0],'r',alpha=frac) #
ed = error_fc[3,:,:]/error_fc[3,:,0].reshape(n_ens,1)
print(np.shape(ed))
axes[3,1].plot(time_vec, ed[:,:].mean(axis=0),'k',linewidth=2.)
axes[3,1].plot(time_vec,2*np.ones(len(time_vec)),'k--',linewidth=2.)
axes[3,1].set_ylabel('$v$',fontsize=18)
axes[3,1].set_xlabel('Assim. time $T$',fontsize=14)

name = '/err_evol_ens_T'+str(T0)+'.pdf'
f_name = str(figsdir+name)
plt.savefig(f_name)
print(' ')
print(' *** %s saved to %s' %(name,figsdir))
print(' ')

print(' ')
print(' CALCULATE : Error-doubling times...')
print(' ')
err_doub = np.empty((Neq,n_ens))
for jj in range(0,Neq):
    for ii in range(0,n_ens):
        try:
            err_doub[jj,ii] = np.where(error_fc[jj,ii,:]/error_fc[jj,ii,0]>2.)[0][0]
        except:
            err_doub[jj,ii] = float('nan')
print(err_doub[0,:])
print(err_doub[1,:])
print(err_doub[2,:])
print(err_doub[3,:])
print(' ')
print(' SAVE : Error-doubling times...')
print(' ')
np.save(str(dirn+'/err_doub_T'+str(T0)),err_doub)
