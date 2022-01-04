#######################################################################
### This script contains a collection of functions used in the EnKF ###
#######################################################################

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
from builtins import range
import numpy as np

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################
from obs_oper import obs_oper, obs_oper_nl, jacobian, Gain_modens_HM, dHdxK
from isen_func import interp_sig2etab, dMdsig, M_int

##################################################################
# FUNCTIONS
##################################################################

### Function to generate the nature run trajectory

def generate_truth(U_tr_array, Nk_tr, tr_grid, Neq, cfl, assim_time, tmax, dtmeasure, f_path_name, R, k, theta1, theta2, eta0, g, Z0, U_scale, sig_r, sig_c, cc2, alpha2, beta, Ro, U_tr_rel, tau_rel, h5_file_data):
    
    from f_ismodRSW import time_step, step_forward_ismodRSW

    Kk_tr = tr_grid[0] 
    x_tr = tr_grid[1]
    
    tn = 0.0
    tmeasure = dtmeasure

    U_tr = U_tr_array[:,:,0]
 
    ### Convection threshold
    etab_c = interp_sig2etab(sig_c,h5_file_data)
    Mc = M_int(etab_c,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
 
    print(' ')
    print('Integrating forward from t =', tn, 'to', tmax,'...')
    print(' ')
    
    index = 1 # for U_tr_array (start from 1 as 0 contains IC).
    while tn < tmax:

        etab = interp_sig2etab(U_tr[0,:],h5_file_data)
        dM_dsig = dMdsig(etab,U_tr[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
        M = M_int(etab,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)

        dt = time_step(U_tr,Kk_tr,cfl,dM_dsig,cc2,beta)
        tn = tn + dt

        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12
                 
        U_tr = step_forward_ismodRSW(U_tr,U_tr_rel,dt,tn,Nk_tr,Neq,Kk_tr,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,tau_rel)

        if tn > tmeasure:
            U_tr_array[:,:,index] = U_tr
            print('*** STORE TRUTH at observing time = ',tmeasure,' ***')
            tmeasure = tmeasure + dtmeasure
            index = index + 1
            
    np.save(f_path_name,U_tr_array)
    
    print('* DONE: truth array saved to:', f_path_name, ' with shape:', np.shape(U_tr_array), ' *')
        
    return U_tr_array    

##################################################################
### Function to compute the Gaspari-Cohn taper function for covariance localisation (from Jeff Whitaker's github: https://github.com/jswhit/)

def gaspcohn(r):
    # Gaspari-Cohn taper function
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
### Function to assemble the Gaspari-Cohn matrix using the taper function

def gaspcohn_matrix(loc_rho,Nk_fc,Neq):
    
    rr = np.arange(0,loc_rho,loc_rho/Nk_fc) 
    vec = gaspcohn(rr)
    
    rho = np.zeros((Nk_fc,Nk_fc))
    
    for i in range(Nk_fc):
        for j in range(Nk_fc):
            rho[i,j] = vec[np.min([abs(i-j),abs(i+Nk_fc-j),abs(i-Nk_fc-j)])]
    
    rho = np.tile(rho, (Neq,Neq))
    
    return rho    

####################################################################
### Function to calculate the square root of the Gaspari-Cohn matrix (from Jeff Whitaker's github: https://github.com/jswhit/L96)
def gaspcohn_sqrt_matrix(rho,n_d):

    evals, eigs = np.linalg.eigh(rho)
    evals = np.where(evals > 1.e-10, evals, 1.e-10)
    evalsum = evals.sum()

    neig = 0
    frac = 0.0
    thresh = 0.99

    while frac < thresh:
        frac = evals[n_d-neig-1:n_d].sum()/evalsum
        neig += 1

    zz = (eigs*np.sqrt(evals/frac)).T
    z = zz[n_d-neig:n_d,:]

    return z.T

##################################################################
### ANALYSIS STEP with linearisation of the observation operator

def analysis_step_enkf_lin(U_fc, U_tr, Y_obs, tmeasure, dtmeasure, index, pars, h5_file_data):

    '''
        INPUTS
        U_fc: ensemble trajectories in U space, shape (Neq,Nk_fc,n_ens)
        U_tr: truth trajectory in U space, shape (Neq,Nk_tr,Nmeas+1)
        Y_obs: observations
        tmeasure: time of assimilation
        dtmeasure: duration of forecast step
        index: integer indicating time of assimilation 
        pars: vector of parameters relating to DA (e.g. relaxation and localisation)
        h5_file_data: look-up table to convert pseudo-density into non-dimensional pressure 
    '''
   
    print(' ')
    print('----------------------------------------------')
    print('------------ ANALYSIS STEP: START ------------')
    print('----------------------------------------------')
    print(' ')
   
    pars_enda = pars[4]
    rtpp = pars_enda[0]
    rtps = pars_enda[1]
    loc = pars_enda[2]
    add_inf = pars_enda[3]
    sat_vel = pars[9]
    ob_noise = pars[15] # 4-vector of obs noise
    n_ens = pars[2] 
    dres = pars[8]
    Nk_fc = pars[0] 
    Neq = pars[13]
    n_d = pars[12]
    n_obs = pars[6]
    n_obs_grnd = pars[11]
    n_obs_sat = pars[10]
    k = pars[16] 
    Rgas = pars[17]
    theta1 = pars[18]
    theta2 = pars[19]
    eta0 = pars[20]
    Z0 = pars[21]
    g = pars[22]
    U_scale = pars[23] 
    sat_obs_mask = pars[24]
    sigu_obs_mask = pars[25]
    sigv_obs_mask = pars[26]
    sigr_obs_mask = pars[27]
    sig_c = pars[28]
    sig_r = pars[29]
    sat_init_pos = pars[30]
    obs_T_d = pars[31]
    obs_u_d = pars[32]
    obs_v_d = pars[33]
    obs_r_d = pars[34]
    n_obs_T = pars[35]
    n_obs_u = pars[36]
    n_obs_v = pars[37]
    n_obs_r = pars[38]
    sigT_obs_mask = pars[39]    
 
    print(' ')
    print('--------- ANALYSIS: EnKF ---------')
    print(' ')
    print('Assimilation time = ', tmeasure)
    print('Number of ensembles = ', n_ens)
   
    # project truth onto forecast grid so that U and U_tr are the same dimension
    U_tmp = np.empty([Neq,Nk_fc])
    for i in range(0,Nk_fc):
        U_tmp[:,i] = U_tr[:,i*dres,index+1]
    U_tr = U_tmp
   
    # for assimilation, work with [sig,u,v,r]
    U_fc[1:,:,:] = U_fc[1:,:,:]/U_fc[0,:,:]
    U_tr[1:,:] = U_tr[1:,:]/U_tr[0,:]
   
    X_tr = U_tr.flatten()
    X_tr = X_tr.T
   
    # state matrix (flatten the array)
    X = np.empty([n_d,n_ens])
    for N in range(0,n_ens):
        X[:,N] = U_fc[:,:,N].flatten()
 
    # make pseudo-obs by defining observation operator and adding perturbations
    # observation operator
    H, row_vec, sat_pos = obs_oper(n_obs, n_obs_sat, sat_init_pos, n_obs_grnd, obs_T_d, obs_u_d, obs_v_d, obs_r_d, n_d, Nk_fc, sat_vel, index)
   
    # Add observation perturbations to each member, ensuring that the
    # observation perturbations have zero mean over all members to
    # avoid perturbing the ensemble mean. Do not apply when rtpp factor is 0.5
    # as Sakov and Oke (2008) results are equivalent to saying that rtpp 0.5
    # gives a deterministic ensemble Kalman filter in which perturbation
    # observations should not be applied.
    ob_noise = np.repeat(ob_noise,[n_obs_sat,n_obs_T,n_obs_u,n_obs_v,n_obs_r])
   
    if rtpp != 0.5:
        obs_pert = ob_noise[:,None]*np.random.randn(n_obs,n_ens)
        obs_pert_mean = np.mean(obs_pert, axis=1)
        obs_pert -= np.repeat(obs_pert_mean, n_ens).reshape(n_obs, n_ens)
        print('obs_pert shape =', np.shape(obs_pert))
        Y_obs_per = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens) + obs_pert
    else: 
        Y_obs_per = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens)   
   
    #### CALCULATE KALMAN GAIN, INNOVATIONS, AND ANALYSIS STATES ####
    ONE = np.ones([n_ens,n_ens])
    ONE = ONE/n_ens # NxN array with elements equal to 1/N
    Xbar = np.repeat(X.mean(axis=1), n_ens).reshape(n_d, n_ens) 
    Xdev = X - Xbar # deviations
   
    # masks for locating model variables in state vector
    sig_mask = list(range(0,Nk_fc))
    sigu_mask = list(range(Nk_fc,2*Nk_fc))
    sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
    sigr_mask = list(range(3*Nk_fc,4*Nk_fc))
    HX, HdBsat = obs_oper_nl(H,X,n_ens,k,n_obs,n_obs_sat,sig_mask,sig_c,sig_r,Rgas,theta1,theta2,eta0,g,Z0,U_scale,h5_file_data) 
    # construct localisation matrix rho based on Gaspari Cohn function
    loc_rho = loc # loc_rho is form of lengthscale.
    rho = gaspcohn_matrix(loc_rho,Nk_fc,Neq)
    print('loc matrix rho shape: ', np.shape(rho))
   
    # compute innovation d = Y-H*X
    D = Y_obs_per - HX
    # construct K and R
    R = ob_noise*ob_noise*np.identity(n_obs) # obs cov matrix
    HKd = np.empty((n_obs,n_ens))
    HKtr = np.empty(n_ens)
    Xan = np.empty((n_d,n_ens))
 
    # covariance matrix
    for i in range(0,n_ens):
        Pf = np.matmul(np.delete(Xdev,i,1), np.delete(Xdev,i,1).T)
        Pf = Pf/(n_ens-2)
        J = jacobian(HdBsat[:,i],H,index,Nk_fc*dres,k,dres,sat_vel,n_obs_sat,sat_init_pos)
        Ktemp = np.matmul(J,np.matmul(rho * Pf,J.T)) + R # H B H^T + R
        Ktemp = np.linalg.inv(Ktemp) # [H B H^T + R]^-1
        K = np.matmul(np.matmul(rho * Pf, J.T), Ktemp) # (rho Pf)H^T [H (rho Pf) H^T + R]^-1
        Xan[:,i] = X[:,i] + np.matmul(K,D[:,i])
        HK = np.matmul(H,K)
        HKd[:,i] = np.diag(HK)
        HKtr[i] = np.trace(HK)
   
    ### Relaxation to Prior Perturbation (RTPP) - Zhang et al. (2004)
    if rtpp != 0.0: # relax to ensemble
        print('RTPP factor =', rtpp)
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Xandev = (1 - rtpp) * Xandev + rtpp * Xdev
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPP relaxation applied')
    ### Relaxation to Prior Spread (RTPS) - Whitaker and Hamill (2012)   
    if rtps != 0.0: # relax the ensemble
        print('RTPS factor =', rtps)
        Pf = np.matmul(Xdev, Xdev.T)           
        Pf = Pf / (n_ens - 1)
        sigma_b = np.sqrt(np.diagonal(Pf))
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Pa = np.matmul(Xandev, Xandev.T) 
        Pa = Pa / (n_ens - 1)
        sigma_a = np.sqrt(np.diagonal(Pa))
        alpha = 1 - rtps + rtps * sigma_b / sigma_a
        print("Min/max RTPS inflation factors = ", np.min(alpha), np.max(alpha))
        Xandev = Xandev * alpha[:, None]
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPS relaxation applied')
    # if r < 0, set to zero:
    r = Xan[sigr_mask,:]
    r[r < 0.] = 0.
    Xan[sigr_mask,:] = r
   
    # if h < 0, set to epsilon:
    h = Xan[sig_mask,:]
    h[h < 0.] = 1e-3
    Xan[sig_mask,:] = h
   
    # transform from X to U for next integration (in u, hu, hr coordinates)
    U_an = np.empty((Neq,Nk_fc,n_ens))
    Xan[sigu_mask,:] = Xan[sigu_mask,:] * Xan[sig_mask,:]
    Xan[sigv_mask,:] = Xan[sigv_mask,:] * Xan[sig_mask,:]
    Xan[sigr_mask,:] = Xan[sigr_mask,:] * Xan[sig_mask,:]
    for N in range(0,n_ens):
        U_an[:,:,N] = Xan[:,N].reshape(Neq,Nk_fc)
   
    # now inflated, transform back to x = (h,u,r) for saving and later plotting
    Xan[sigu_mask,:] = Xan[sigu_mask,:]/Xan[sig_mask,:]
    Xan[sigv_mask,:] = Xan[sigv_mask,:]/Xan[sig_mask,:]
    Xan[sigr_mask,:] = Xan[sigr_mask,:]/Xan[sig_mask,:]
   
    print(' ')
    print('--------- CHECK SHAPE OF MATRICES: ---------')
    print(' ')
    print('U_fc shape   :', np.shape(U_fc))
    print('U_tr shape   :', np.shape(U_tr))
    print('X_truth shape:', np.shape(X_tr), '( NOTE: should be', n_d,' by 1 )')
    print('X shape      :', np.shape(X), '( NOTE: should be', n_d,'by', n_ens,')')
    print('Xbar shape   :', np.shape(Xbar))
    print('Xdev shape   :', np.shape(Xdev))
    print('Pf shape     :', np.shape(Pf), '( NOTE: should be n by n square for n=', n_d,')')
    print('H shape      :', np.shape(H), '( NOTE: should be', n_obs,'by', n_d,')')
    print('K shape      :', np.shape(K), '( NOTE: should be', n_d,'by', n_obs,')')
    print('Y_obs shape  :', np.shape(Y_obs))
    print('Xan shape    :', np.shape(Xan), '( NOTE: should be the same as X shape)')
    print('U_an shape   :', np.shape(U_an), '( NOTE: should be the same as U_fc shape)')
   
   
    ## observational influence diagnostics
    print(' ')
    print('--------- OBSERVATIONAL INFLUENCE DIAGNOSTICS:---------')
    print(' ')
    print(' Benchmark: global NWP has an average OI of ~0.18... ')
    print(' ... high-res. NWP less clear but should be 0.15 - 0.4')
    print('Check below: ')
    OI_vec = np.zeros(Neq+2)
    OI_vec[0] = np.mean(HKtr,0)/n_obs
    HKd_mean = np.mean(HKd,1)
    if(n_obs_sat>0): OI_vec[1] = np.sum(HKd_mean[sat_obs_mask])/n_obs
    if(n_obs_T>0): OI_vec[2] = np.sum(HKd_mean[sigT_obs_mask])/n_obs
    if(n_obs_u>0): OI_vec[3] = np.sum(HKd_mean[sigu_obs_mask])/n_obs
    if(n_obs_v>0): OI_vec[4] = np.sum(HKd_mean[sigv_obs_mask])/n_obs
    if(n_obs_r>0): OI_vec[5] = np.sum(HKd_mean[sigr_obs_mask])/n_obs

    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print('OI =', OI_vec[0])
    print('OI check = ', np.sum(HKd_mean)/n_obs)
    print('OI_sat =', OI_vec[1])
    print('OI_sigT =', OI_vec[2])
    print('OI_sigu =', OI_vec[3])
    print('OI_sigv =', OI_vec[4])
    print('OI_sigr =', OI_vec[5])
    print(' ')
    print(' ')
    print('----------------------------------------------')
    print('------------- ANALYSIS STEP: END -------------')
    print('----------------------------------------------')
    print(' ')
   
    return U_an, U_fc, X, X_tr, Xan, OI_vec

##################################################################
### ANALYSIS STEP with modulated ensemble and non-linear observation operator

def analysis_step_enkf_nl(U_fc, U_tr, Y_obs, tmeasure, dtmeasure, index, pars, h5_file_data, dirn):

    '''
        INPUTS
        U_fc: ensemble trajectories in U space, shape (Neq,Nk_fc,n_ens)
        U_tr: truth trajectory in U space, shape (Neq,Nk_tr,Nmeas+1)
        Y_obs: observations
        tmeasure: time of assimilation
        dtmeasure: duration of forecast step
        index: integer indicating time of assimilation 
        pars: vector of parameters relating to DA (e.g. relaxation and localisation)
        h5_file_data: look-up table to convert pseudo-density into non-dimensional pressure 
        dirn: output directory to save debug data
    '''
   
    print(' ')
    print('----------------------------------------------')
    print('------------ ANALYSIS STEP: START ------------')
    print('----------------------------------------------')
    print(' ')
   
    pars_enda = pars[4]
    rtpp = pars_enda[0]
    rtps = pars_enda[1]
    loc = pars_enda[2]
    add_inf = pars_enda[3]
    sat_vel = pars[9]
    ob_noise = pars[15] # 4-vector of obs noise
    n_ens = pars[2] 
    dres = pars[8]
    Nk_fc = pars[0] 
    Neq = pars[13]
    n_d = pars[12]
    n_obs = pars[6]
    n_obs_grnd = pars[11]
    n_obs_sat = pars[10]
    k = pars[16] 
    Rgas = pars[17]
    theta1 = pars[18]
    theta2 = pars[19]
    eta0 = pars[20]
    Z0 = pars[21]
    g = pars[22]
    U_scale = pars[23] 
    sat_obs_mask = pars[24]
    sigu_obs_mask = pars[25]
    sigv_obs_mask = pars[26]
    sigr_obs_mask = pars[27]
    sig_c = pars[28]
    sig_r = pars[29]
    sat_init_pos = pars[30]
    obs_T_d = pars[31] 
    obs_u_d = pars[32]
    obs_v_d = pars[33]
    obs_r_d = pars[34]
    n_obs_T = pars[35]
    n_obs_u = pars[36]
    n_obs_v = pars[37]
    n_obs_r = pars[38]
    sigT_obs_mask = pars[39]

    print(' ')
    print('--------- ANALYSIS: EnKF ---------')
    print(' ')
    print('Assimilation time = ', tmeasure)
    print('Number of ensembles = ', n_ens)
   
    # project truth onto forecast grid so that U and U_tr are the same dimension
    U_tmp = np.empty([Neq,Nk_fc])
    for i in range(0,Nk_fc):
        U_tmp[:,i] = U_tr[:,i*dres,index+1]
    U_tr = U_tmp
   
    # for assimilation, work with [sig,u,v,r]
    U_fc[1:,:,:] = U_fc[1:,:,:]/U_fc[0,:,:]
    U_tr[1:,:] = U_tr[1:,:]/U_tr[0,:]
   
    X_tr = U_tr.flatten()
    X_tr = X_tr.T
   
    # state matrix (flatten the array)
    X = np.empty([n_d,n_ens])
    for N in range(0,n_ens):
        X[:,N] = U_fc[:,:,N].flatten()
 
    # make pseudo-obs by defining observation operator and adding perturbations
    # observation operator
    H, row_vec, sat_pos = obs_oper(n_obs, n_obs_sat, sat_init_pos, n_obs_grnd, obs_T_d, obs_u_d, obs_v_d, obs_r_d, n_d, Nk_fc, sat_vel, index)
  
    np.save(str(dirn+'/H_obs_oper.npy'),H) # save linear observation operator for debug

    # Add observation perturbations to each member, ensuring that the
    # observation perturbations have zero mean over all members to
    # avoid perturbing the ensemble mean. Do not apply when rtpp factor is 0.5
    # as Sakov and Oke (2008) results are equivalent to saying that rtpp 0.5
    # gives a deterministic ensemble Kalman filter in which perturbation
    # observations should not be applied.
    ob_noise = np.repeat(ob_noise,[n_obs_sat,n_obs_T,n_obs_u,n_obs_v,n_obs_r])
   
    if rtpp != 0.5:
        obs_pert = ob_noise[:,None]*np.random.randn(n_obs,n_ens)
        obs_pert_mean = np.mean(obs_pert, axis=1)
        obs_pert -= np.repeat(obs_pert_mean, n_ens).reshape(n_obs, n_ens)
        print('obs_pert shape =', np.shape(obs_pert))
        Y_obs_per = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens) + obs_pert
    else: 
        Y_obs_per = np.repeat(Y_obs[:, index], n_ens).reshape(n_obs, n_ens)   

    #### CALCULATE KALMAN GAIN, INNOVATIONS, AND ANALYSIS STATES ####
    ONE = np.ones([n_ens,n_ens])
    ONE = ONE/n_ens # NxN array with elements equal to 1/N
    Xbar = np.repeat(X.mean(axis=1), n_ens).reshape(n_d, n_ens) 
    Xdev = X - Xbar # deviations
   
    # masks for locating model variables in state vector
    sig_mask = list(range(0,Nk_fc))
    sigu_mask = list(range(Nk_fc,2*Nk_fc))
    sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
    sigr_mask = list(range(3*Nk_fc,4*Nk_fc))
    HX, HdBsat = obs_oper_nl(H,X,n_ens,k,n_obs,n_obs_sat,sig_mask,sig_c,sig_r,Rgas,theta1,theta2,eta0,g,Z0,U_scale,h5_file_data) 
    np.save(str(dirn+'/H_obs_oper_nl.npy'),H) # save nonlinear observation operator for debug

    # construct localisation matrix rho based on Gaspari Cohn function
    loc_rho = loc # loc_rho is form of lengthscale.
    rho = gaspcohn_matrix(loc_rho,Nk_fc,Neq)
    print('loc matrix rho shape: ', np.shape(rho))
   
    # compute innovation d = Y-H*X
    D = Y_obs_per - HX
    # construct K and R
    R = ob_noise*ob_noise*np.identity(n_obs) # obs cov matrix
    HK = np.zeros((n_obs,n_obs))
    HKd = np.empty((n_obs,n_ens))
    HKtr = np.empty(n_ens)
    Xan = np.empty((n_d,n_ens))

    S = np.zeros((n_d,n_ens))

    # covariance matrix
    for i in range(0,n_ens):
        K = Gain_modens_HM(rho,n_ens-1,n_d,Xbar[:,0],np.delete(Xdev,i,1),sig_mask,k,H,sig_c,sig_r,n_obs,n_obs_sat,R,h5_file_data)
        S[:,i] = np.matmul(K,D[:,i])
        Xan[:,i] = X[:,i] + np.matmul(K,D[:,i])
        if(n_obs_sat+n_obs_T>0): 
           HK[:(n_obs_sat+n_obs_T),:] = dHdxK(Xan[:,i],sig_mask,sig_r,sig_c,n_obs,H[:(n_obs_sat+n_obs_T),:],K,n_obs_sat,n_obs_T,Rgas,k,theta1,theta2,eta0,g,Z0,U_scale,h5_file_data)
        HK[(n_obs_sat+n_obs_T):,:] = np.matmul(H[(n_obs_sat+n_obs_T):,:],K)
        HKd[:,i] = np.diag(HK)
        HKtr[i] = np.trace(HK)
   
    ### Relaxation to Prior Perturbation (RTPP) - Zhang et al. (2004)
    if rtpp != 0.0: # relax to ensemble
        print('RTPP factor =', rtpp)
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Xandev = (1 - rtpp) * Xandev + rtpp * Xdev
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPP relaxation applied')
    ### Relaxation to Prior Spread (RTPS) - Whitaker and Hamill (2012)   
    if rtps != 0.0: # relax the ensemble
        print('RTPS factor =', rtps)
        Pf = np.matmul(Xdev, Xdev.T)           
        Pf = Pf / (n_ens - 1)
        sigma_b = np.sqrt(np.diagonal(Pf))
        Xanbar = np.repeat(Xan.mean(axis=1), n_ens).reshape(n_d, n_ens)
        Xandev = Xan - Xanbar  # analysis deviations
        Pa = np.matmul(Xandev, Xandev.T) 
        Pa = Pa / (n_ens - 1)
        sigma_a = np.sqrt(np.diagonal(Pa))
        alpha = 1 - rtps + rtps * sigma_b / sigma_a
        print("Min/max RTPS inflation factors = ", np.min(alpha), np.max(alpha))
        Xandev = Xandev * alpha[:, None]
        Xan = Xandev + Xanbar # relaxed analysis ensemble
    else:
        print('No RTPS relaxation applied')
    # if r < 0, set to zero:
    r = Xan[sigr_mask,:]
    r[r < 0.] = 0.
    Xan[sigr_mask,:] = r
   
    # if h < 0, set to epsilon:
    h = Xan[sig_mask,:]
    h[h < 0.] = 1e-3
    Xan[sig_mask,:] = h
   
    # transform from X to U for next integration (in u, hu, hr coordinates)
    U_an = np.empty((Neq,Nk_fc,n_ens))
    Xan[sigu_mask,:] = Xan[sigu_mask,:] * Xan[sig_mask,:]
    Xan[sigv_mask,:] = Xan[sigv_mask,:] * Xan[sig_mask,:]
    Xan[sigr_mask,:] = Xan[sigr_mask,:] * Xan[sig_mask,:]
    for N in range(0,n_ens):
        U_an[:,:,N] = Xan[:,N].reshape(Neq,Nk_fc)
   
    # now inflated, transform back to x = (h,u,r) for saving and later plotting
    Xan[sigu_mask,:] = Xan[sigu_mask,:]/Xan[sig_mask,:]
    Xan[sigv_mask,:] = Xan[sigv_mask,:]/Xan[sig_mask,:]
    Xan[sigr_mask,:] = Xan[sigr_mask,:]/Xan[sig_mask,:]
   
    print(' ')
    print('--------- CHECK SHAPE OF MATRICES: ---------')
    print(' ')
    print('U_fc shape   :', np.shape(U_fc))
    print('U_tr shape   :', np.shape(U_tr))
    print('X_truth shape:', np.shape(X_tr), '( NOTE: should be', n_d,' by 1 )')
    print('X shape      :', np.shape(X), '( NOTE: should be', n_d,'by', n_ens,')')
    print('Xbar shape   :', np.shape(Xbar))
    print('Xdev shape   :', np.shape(Xdev))
    print('Pf shape     :', np.shape(Pf), '( NOTE: should be n by n square for n=', n_d,')')
    print('H shape      :', np.shape(H), '( NOTE: should be', n_obs,'by', n_d,')')
    print('K shape      :', np.shape(K), '( NOTE: should be', n_d,'by', n_obs,')')
    print('Y_obs shape  :', np.shape(Y_obs))
    print('Xan shape    :', np.shape(Xan), '( NOTE: should be the same as X shape)')
    print('U_an shape   :', np.shape(U_an), '( NOTE: should be the same as U_fc shape)')
   
   
    ## observational influence diagnostics
    print(' ')
    print('--------- OBSERVATIONAL INFLUENCE DIAGNOSTICS:---------')
    print(' ')
    print(' Benchmark: global NWP has an average OI of ~0.18... ')
    print(' ... high-res. NWP less clear but should be 0.15 - 0.4')
    print('Check below: ')
    OI_vec = np.zeros(Neq+2)
    OI_vec[0] = np.mean(HKtr,0)/n_obs
    HKd_mean = np.mean(HKd,1)
    if(n_obs_sat>0): OI_vec[1] = np.sum(HKd_mean[sat_obs_mask])/n_obs
    if(n_obs_T>0): OI_vec[2] = np.sum(HKd_mean[sigT_obs_mask])/n_obs
    if(n_obs_u>0): OI_vec[3] = np.sum(HKd_mean[sigu_obs_mask])/n_obs
    if(n_obs_v>0): OI_vec[4] = np.sum(HKd_mean[sigv_obs_mask])/n_obs
    if(n_obs_r>0): OI_vec[5] = np.sum(HKd_mean[sigr_obs_mask])/n_obs

    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print('OI =', OI_vec[0])
    print('OI check = ', np.sum(HKd_mean)/n_obs)
    print('OI_sat =', OI_vec[1])
    print('OI_sigT =', OI_vec[2])
    print('OI_sigu =', OI_vec[3])
    print('OI_sigv =', OI_vec[4])
    print('OI_sigr =', OI_vec[5])
    print(' ')
    print(' ')
    print('----------------------------------------------')
    print('------------- ANALYSIS STEP: END -------------')
    print('----------------------------------------------')
    print(' ')
   
    return U_an, U_fc, X, X_tr, Xan, OI_vec

