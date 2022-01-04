#######################################################################
### This script generates a Q matrix according to the parameters in the configuration file

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import importlib
import sys
import h5py
from scipy import linalg

##################################################################
# HANDLE WARNINGS AS ERRORS
##################################################################
from f_ismodRSW import make_grid, step_forward_ismodRSW, time_step
from isen_func import interp_sig2etab, dMdsig, M_int

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
Nk_fc = config.Nk_fc
L = config.L
V = config.V
Nmeas = config.Nmeas
Neq = config.Neq
dres = config.dres
cfl_fc = config.cfl_fc
sig_c = config.sig_c
sig_r = config.sig_r
assim_time = config.assim_time
dtmeasure = config.dtmeasure
cc2 = config.cc2
beta = config.beta
alpha2 = config.alpha2
Ro = config.Ro
U_relax = config.U_relax
tau_rel = config.tau_rel
g = config.g
R = config.R
theta1 = config.theta1
theta2 = config.theta2
eta0 = config.eta0
Z0 = config.Z0
U_scale = config.U_scale
k = config.k
model_noise = config.model_noise
Nhr = config.Nhr
Q_FUNC = config.Q_FUNC
rMODNOISE = config.rMODNOISE
sigMODNOISE = config.sigMODNOISE
ass_freq = config.ass_freq

#################################################################

# Q-matrix from pre-defined model noise (diagonal)
def Q_predef():
    var_sig = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[0]
    var_u = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[1]
    var_v = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[2]
    var_r = list(map(lambda x: np.repeat(x**2, Nk_fc), model_noise))[3]
    Q = np.diag(np.concatenate((var_sig,var_u,var_v,var_r)))

    return Q

#################################################################
# Cycle over truth trajectory times.  Use each value as the initial condition
# for both the truth (obtained from X_tr at the next timestep) and a
# lower-resolution forecast.
def Q_nhr():
    B = np.load(str(outdir + '/B_tr.npy')) # truth
    U_tr = np.load(str(outdir + '/U_tr_array_2xres_'+ass_freq+'.npy')) # truth
    ### LOAD LOOK-UP TABLE
    h5_file = h5py.File('inversion_tables/sigma_eta_theta2_291_theta1_311_eta0_0.48_Z0_6120_k_0.29.hdf','r')
    h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]
    fc_grid = make_grid(Nk_fc, L) # forecast
    Kk_fc = fc_grid[0]
    xc = fc_grid[2]
    nsteps = Nmeas
    X = np.zeros([Neq * Nk_fc, nsteps])
    U = np.copy(U_tr[:, 0::dres, :]) # Sample truth at forecast resolution
    ### Relaxation solution ###
    U_rel = U_relax(Neq,Nk_fc,L,V,xc,U[:,:,0])
    etab_c = interp_sig2etab(sig_c,h5_file_data) 
    Mc = M_int(etab_c,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale) 
    for T in range(nsteps):
        tn = assim_time[T]
        tmeasure = tn+Nhr*dtmeasure
        print('*** Integrating between time:', tn, ' and: ', tmeasure, ' ***')
        U_fc = np.copy(U[:, :, T])
        while tn < tmeasure:
            etab = interp_sig2etab(U_fc[0,:],h5_file_data)
            dM_dsig = dMdsig(etab,U_fc[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
            M = M_int(etab,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
            dt = time_step(U_fc, Kk_fc, cfl_fc, dM_dsig, cc2, beta) # compute stable time step
            tn = tn + dt
            if tn > tmeasure:
                dt = dt - (tn - tmeasure) + 1e-12
                tn = tmeasure + 1e-12
            U_fc = step_forward_ismodRSW(U_fc,U_rel,dt,tn,Nk_fc,Neq,Kk_fc,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,tau_rel)
        print('*** Storing the error at time: ', tmeasure, ' ***')
        X[:, T] = U[:, :, T+Nhr].flatten() - U_fc.flatten()
    print("Means of proxy error components = ", np.mean(U[:, :, 1:(nsteps+1)] - np.repeat(U_fc, nsteps).reshape(Neq, Nk_fc, nsteps), axis=(1, 2)))

    # Compute the covariance matrix.
    Q = np.cov(X, bias=False)
    # Extrapolate diagonal of Q
    Q_diag = np.array(Q.diagonal())
    # Pose the covariance of r to 0
    if(rMODNOISE==0):
        Q_diag[3*Nk_fc:] = 0.0
    if(sigMODNOISE==0):
        Q_diag[:Nk_fc] = 0.0

    # Return a diagonal matrix
    Q = np.diag(Q_diag)

    return Q

##################################################################

f_path_Q = str(outdir+'/Qmatrix.npy') 

# Load or generate Q matrix according to the choice made in the config file 
try:
    print(' *** Loading Q matrix *** ')
    Q = np.load(f_path_Q)
except:
    print(' *** Generating Q matrix *** ')
    Q = eval(Q_FUNC)
    print(("Min. and max. variances", np.amin(Q.diagonal()), np.amax(Q.diagonal())))
    np.save(str(outdir + '/Qmatrix'), Q)
