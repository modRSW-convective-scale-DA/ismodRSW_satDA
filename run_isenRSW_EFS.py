#######################################################################
# Ensemble forecasts for 1.5D SWEs with rain variable and topography
#######################################################################

'''
    25.09.2016
    
    Script runs ensemble forecasts, initialised from analysed ensembles at a given time.
    
    Specify: directory, ijk, T0, Tfc
    
    '''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import os
import errno
import sys
import itertools
import h5py
import importlib.util
import numpy as np
import multiprocessing as mp
from datetime import datetime

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################

from f_isenRSW import make_grid, time_step, ens_forecast
from create_readme import create_readme
from isen_func import interp_sig2etab, M_int

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
L = config.L
V = config.V
Nk_fc = config.Nk_fc
dres = config.dres
ic = config.ic
tmeasure = config.tmeasure
n_d = config.n_d
n_ens = config.n_ens
Neq = config.Neq
cfl_fc = config.cfl_fc
dtmeasure = config.dtmeasure
sig_c = config.sig_c
sig_r = config.sig_r
Ro = config.Ro
cc2 = config.cc2
alpha2 = config.alpha2
beta = config.beta
g = config.g
R = config.R
kgas = config.k
tau_rel = config.tau_rel
eta0 = config.eta0
theta1 = config.theta1
theta2 = config.theta2
Z0 = config.Z0
U_scale = config.U_scale
NIAU = config.NIAU
U_relax = config.U_relax
tau_rel = config.tau_rel

###################################################################
# Read experiment index from command line and split it into indeces
###################################################################

n_job = int(sys.argv[2])-1
indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))
i = indices[n_job][0]
j = indices[n_job][1]
k = indices[n_job][2]
l = indices[n_job][3]

##################################################################
# make EDT directory (if it doesn't already exist)

dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]),
                                                 str(add_inf[j]), str(rtpp[k]),
                                                 str(rtps[l]))
dirnEDT = str(dirn+'/EDT')

try:
    os.makedirs(dirnEDT)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

##################################################################
# Mesh generation for forecasts
##################################################################

fc_grid =  make_grid(Nk_fc,L) # forecast

Kk_fc = fc_grid[0]
x_fc = fc_grid[1]
xc_fc = fc_grid[2]

# masks for locating model variables in state vector
sig_mask = list(range(0,Nk_fc))
sigu_mask = list(range(Nk_fc,2*Nk_fc))
sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

#LOAD SAVED DATA
X_array = np.load(str(dirn+'/Xan_array.npy')) # load ANALYSIS ensembles to sample ICs
X_tr = np.load(str(dirn+'/X_tr_array.npy'))
B = np.load(str(dirn+'/B.npy'))
#X_EFS = np.load(str(dirn+'/X_EFS_array_T24.npy'))

### LOAD TRUTH, OBSERVATIONS AND OBSERVATION OPERATOR ###

f_path_name = str(outdir+'/U_tr_array_2xres_1h.npy')

try:
    ' *** Loading truth trajectory... *** '
    U_tr_array = np.load(f_path_name)
except:
    print(' Failed to find the truth trajectory: run create_truth+obs.py first')

T = int(sys.argv[3])-1 # to locate IC from saved data
tn = 0.0
Tfc = 36 # length of forecast (hrs)
tmax = Tfc*tmeasure
assim_time = np.linspace(tn,tmax,Tfc+1)
print(assim_time)

X0 = X_array[:,:,T]
X0_tr = X_tr[:,:,T]

X_fc_array = np.empty([n_d,n_ens,Tfc+1])
X_fc_array[:,:,0] = X0

### LOAD LOOK-UP TABLE
h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

### Convection and rain thresholds
etab_c = interp_sig2etab(sig_c,h5_file_data)
Mc = M_int(etab_c,0.,R,kgas,theta1,theta2,eta0,g,Z0,U_scale)

##################################################################
#  Load initial conditions
##################################################################

print(' ')
print('---------------------------------------------------')
print('---------      ICs: load from saved data     ---------')
print('---------------------------------------------------')
print(' ')

X0[sigu_mask,:] = X0[sigu_mask,:]*X0[sig_mask,:]
X0[sigv_mask,:] = X0[sigv_mask,:]*X0[sig_mask,:]
X0[sigr_mask,:] = X0[sigr_mask,:]*X0[sig_mask,:]
U0 = X0.reshape(Neq,Nk_fc,n_ens)
print(np.shape(U0))
X0_tr[sigu_mask,0] = X0_tr[sigu_mask,0]*X0_tr[sig_mask,0]
X0_tr[sigv_mask,0] = X0_tr[sigv_mask,0]*X0_tr[sig_mask,0]
X0_tr[sigr_mask,0] = X0_tr[sigr_mask,0]*X0_tr[sig_mask,0]
U0_tr = X0_tr[:,0].reshape(Neq,Nk_fc)
print(np.shape(U0_tr))

### Define relaxation solution
U_rel = U_relax(Neq,Nk_fc,L,V,xc_fc,U_tr_array[:,0::dres,0])

### Forecast ic 
#U0_fc, B = ic(x_fc,Nk_fc,Neq,H0,L,A,V) # control IC to be perturbed

#U0ens = np.empty([Neq,Nk_fc,n_ens])

#print('Initial ensemble perurbations:')
#print('sig_ic = [sig_h, sig_hu, sig_hr] =', sig_ic)

# Generate initial ensemble
#for jj in range(0,Neq):
#    for N in range(0,n_ens):
#        # add sig_ic to EACH GRIDPOINT
#        U0ens[jj,:,N] = U0_fc[jj,:] + sig_ic[jj]*np.random.randn(Nk_fc)

#U0ens = np.empty([Neq,Nk_fc,n_ens])
#for N in range(0,n_ens):
#    U0ens[:,:,N] = X0[:,N].reshape(Neq,Nk_fc)

#print(' ')
#print('Check ICs...')
#print(' ')

#fig, axes = plt.subplots(3, 1, figsize=(8,8))
#axes[0].plot(xc_fc, U0ens[0,:,:]+B.reshape(len(xc_fc),1), 'b')
#axes[0].plot(xc_fc, U0_tr[0,:]+B.reshape(len(xc_fc),1), 'g')
#axes[0].plot(xc_fc, B, 'k', linewidth=2.)
#axes[0].set_ylabel('$h_0(x)$',fontsize=18)
#axes[0].set_ylim([0,np.max(U0ens[0,:,:])])
#axes[1].plot(xc_fc, U0ens[1,:,:], 'b')
#axes[1].plot(xc_fc, U0_tr[1,:], 'g')
#axes[1].set_ylabel('$hu_0(x)$',fontsize=18)
#axes[2].plot(xc_fc, U0ens[2,:,:], 'b')
#axes[2].plot(xc_fc, U0_tr[2,:], 'g')
#axes[2].set_ylabel('$hr_0(x)$',fontsize=18)
#axes[2].set_xlabel('$x$',fontsize=18)
#plt.interactive(True)
#plt.show() # use block=False?
#exit()
#np.save('test_model/U0ens',U0ens)

##################################################################
#  Integrate ensembles forward in time until obs. is available   #
##################################################################
print(' ')
print('-------------------------------------------------')
print('     ------ ENSEMBLE FORECAST SYSTEM ------      ')
print('-------------------------------------------------')
print(' ')

##if from start...
U = np.copy(U0)
index=0
print('tmax =', tmax)
print('dtmeasure =', dtmeasure)
print('tmeasure = ', tmeasure)

##################################################################
# Load the model error covariance matrix
##################################################################
Q = np.load(str(outdir + '/Qmatrix.npy'))

while tmeasure-dtmeasure <= tmax and index < Tfc:
    print(' ')
    print('----------------------------------------------')
    print('---------- ENSEMBLE FORECAST: START ----------')
    print('----------------------------------------------')
    print(' ')
        
    try:
        os.sched_setaffinity(0, range(0, os.cpu_count()))
        num_cores_use = n_ens
    
        print('Number of cores used:', num_cores_use) 
        print('Starting ensemble integrations from time =', assim_time[index], ' to', assim_time[index+1])
        print(' *** Started: ', str(datetime.now()))
        print(np.shape(U))
                
        ### ADDITIVE INFLATION (moved to precede the forecast) ###
        q = add_inf[j] * np.random.multivariate_normal(np.zeros(n_d), Q, n_ens)
        q_ave = np.mean(q,axis=0)
        q = q - q_ave
        q = q.T

        pool = mp.Pool(processes=num_cores_use)
        mp_out = [pool.apply_async(ens_forecast, args=(N, U, U_rel, tau_rel, q, Neq, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Mc, sig_c, sig_r, cc2, beta, alpha2, Ro, R, kgas, theta1, theta2, eta0, g, Z0, U_scale, h5_file_data)) for N in range(0,n_ens)]
        U = [p.get() for p in mp_out]
        pool.close()
        pool.join()
    
        print(' All ensembles integrated forward from time =', assim_time[index],' to', assim_time[index+1])
        print(' *** Ended: ', str(datetime.now()))
        print(np.shape(U))
    
        U =np.swapaxes(U,0,1)
        U =np.swapaxes(U,1,2)
    
        print(np.shape(U))
    
        print(' ')
        print('----------------------------------------------')
        print('------------- FORECAST STEP: END -------------')
        print('----------------------------------------------')
        print(' ')
        
        ##################################################################
        # save data for calculating crps and error at this time then integrate forward  #
        ##################################################################
    
        # transform to X for saving
        U_tmp = np.copy(U)
        U_tmp[1:,:,:] = U_tmp[1:,:,:]/U_tmp[0,:,:]
        for N in range(0,n_ens):
            X_fc_array[:,N,index+1] = U_tmp[:,:,N].flatten()
    
       # U = UU # update U for next integration

        # on to next assim_time
        index = index + 1
        tmeasure = tmeasure + dtmeasure
    
    except (RuntimeWarning, mp.TimeoutError) as err:
        
        pool.terminate()
        pool.join()
        print(err)
        print('-------------- Forecast failed! --------------')        
        print(' ')
        print('----------------------------------------------')
        print('------------- FORECAST STEP: END -------------')
        print('----------------------------------------------')
        print(' ')

        tmeasure = tmax + dtmeasure
 
np.save(str(dirnEDT+'/X_EFS_array_T'+str(T)),X_fc_array)
print(' *** Data saved in :', dirnEDT)
print(' ')

##################################################################
#                       END OF PROGRAM                           #
##################################################################
