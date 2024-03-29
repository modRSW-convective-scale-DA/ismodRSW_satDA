######################################################################
# Observation Operator for ismodRSW
######################################################################

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
from scipy import signal
from builtins import range
import os
import sys
import numpy as np

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################
from isen_func import interp_sig2etab,dsig_detab

######################################################################
### Observation operator with spatially varying observations: satellite observation moving along the domain from the left to the right plus fixed observations of u,v,r

def obs_oper(n_obs, n_obs_sat, sat_init_pos, n_obs_grnd, obs_T_d, obs_u_d, obs_v_d, obs_r_d, n_d, Nk_fc, sat_vel, T):

    H = np.zeros((n_obs,n_d))
    row_vec_T = list(range(obs_T_d, Nk_fc+1, obs_T_d))
    row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
    row_vec_v = list(range(2*Nk_fc+obs_v_d, 3*Nk_fc+1, obs_v_d))
    row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))
    row_vec = np.array(row_vec_T+row_vec_u+row_vec_v+row_vec_r) 
    sat_pos = ((sat_init_pos*Nk_fc+sat_vel*Nk_fc*(T+1))%Nk_fc).astype(int)
    for i in range(0,n_obs):
        if(i<n_obs_sat): 
            H[i,sat_pos[i]] = 1
        else:
            H[i,row_vec[i-n_obs_sat]-1] = 1

    return H, row_vec, sat_pos

#######################################################################
### Generate radiance observations in a satellite-like way

def obs_generator(U_tmp,Nk_tr,Nk_fc,Kk_tr,dres,Y_obs,ob_noise,sat_vel,n_obs_sat,swathe,n_obs_grnd,n_obs_r,n_d_grnd,obs_T_d,obs_u_d,obs_v_d,obs_r_d,sigr_obs_mask,T,k,sig_c,sig_r,sat_init_pos,h5_file_data):

    import scipy.special as sp

    ### Satellite observations (of radiance)
    X_tr_sat = U_tmp[0, :, T+1]
    sat_pos = ((sat_init_pos*Nk_tr+sat_vel*Nk_tr*(T+1))%Nk_tr).astype(int)
    # Create Gaussian smoother centered on satellite position
    gauss_w = np.zeros((Nk_tr,n_obs_sat))
    sw_len = (swathe/(Kk_tr*500)).astype(int)

    # Generate net 2-layers radiance
    eta2 = interp_sig2etab(X_tr_sat,h5_file_data)
    B2 = eta2**k # radiance coming from bottom layer 
    B1 = (eta2 - X_tr_sat)**k # radiance coming from top layer (eta1 = eta2 - sigma)

    ### Alpha weights
    alpha1 = 0.5-0.5*sp.erf(-95*X_tr_sat+21.5)
    alpha2 = 0.425+0.425*sp.erf(-95*X_tr_sat+21.5)
    alpha3 = 0.5+0.5*sp.erf(-5*X_tr_sat+3)
    alpha4 = 0.5+0.5*sp.erf(-3*X_tr_sat-1.16)

    # Compute net radiance coming from both layers
    Bsat_tr = alpha1*alpha3*B1 + (alpha2+alpha4)*B2

    # Create radiance observation noise
    obs_pert_sat = ob_noise[0:n_obs_sat]*np.random.randn(n_obs_sat)	
    
    # Generate radiance observations
    Bsat = np.zeros(n_obs_sat)
    for i in range(n_obs_sat):
        gauss_w[0:sw_len[i],i] = signal.gaussian(sw_len[i],sw_len[i]/6.)
        gauss_w[:,i] = np.roll(gauss_w[:,i],sat_pos[i]-int(sw_len[i]/2))
        Bsat[i] = np.dot(gauss_w[:,i],Bsat_tr)/sum(gauss_w[:,i]) # apply gaussian filter to spatially varying Bsat_tr
        Y_obs[i,T] = Bsat[i] + obs_pert_sat[i]
 
    ### Ground observations
    X_tr_grnd = U_tmp[:, 0::dres, T+1].flatten()
    X_tr_grnd[:Nk_fc] = B2[0::dres]
    X_tr_grnd = X_tr_grnd.T
    H = np.zeros((n_obs_grnd,n_d_grnd))
    row_vec_T = list(range(obs_T_d, Nk_fc+1, obs_T_d))
    row_vec_u = list(range(Nk_fc+obs_u_d, 2*Nk_fc+1, obs_u_d))
    row_vec_v = list(range(2*Nk_fc+obs_v_d, 3*Nk_fc+1, obs_v_d))
    row_vec_r = list(range(3*Nk_fc+obs_r_d, 4*Nk_fc+1, obs_r_d))
    row_vec = np.array(row_vec_T+row_vec_u+row_vec_v+row_vec_r) 
    for i in range(0,n_obs_grnd):
        H[i,row_vec[i]-1] = 1
    Y_mod = np.dot(H, X_tr_grnd)
    obs_pert_grnd = ob_noise[n_obs_sat:] * np.random.randn(n_obs_grnd)
    Y_obs[n_obs_sat:, T] = Y_mod.flatten() + obs_pert_grnd
    # Reset pseudo-observations with negative h or r to zero.
    if(n_obs_r>0): 
       mask = sigr_obs_mask[np.array(Y_obs[sigr_obs_mask, T] < 0.0)]
       Y_obs[mask, T] = 0.0

    return Y_obs[:,T]

####################################################################################
### Function that deals with the non-linearity of the observation operator and computes its derivative for the jacobian (when used)

def obs_oper_nl(H,X,n_ens,k,n_obs,n_obs_sat,sig_mask,sig_c,sig_r,R,theta1,theta2,eta0,g,Z0,U_scale,h5_file_data):

    import scipy.special as sp   
 
    ### Manipulating X[sig_mask] to get radiance from each layer
    eta2 = interp_sig2etab(X[sig_mask,:],h5_file_data)
    eta1 = interp_sig2etab(X[sig_mask,:],h5_file_data) - X[sig_mask,:]
    B2 = eta2**k
    B1 = eta1**k

    ### Alpha weights
    alpha1 = 0.5-0.5*sp.erf(-95*X[sig_mask,:]+21.5)
    alpha2 = 0.425+0.425*sp.erf(-95*X[sig_mask,:]+21.5)
    alpha3 = 0.5+0.5*sp.erf(-5*X[sig_mask,:]+3)
    alpha4 = 0.5+0.5*sp.erf(-3*X[sig_mask,:]-1.16)

    ### Compute net radiance and replace in X[sig_mask]
    Bsat = B1*alpha3*alpha1 + B2*(alpha2+alpha4)
    Xtmp = np.copy(X)
    Xtmp[sig_mask,:] = B2 # This is for when temperature is observed

    ### Compute the projection of the model onto the observation space
    HX = np.zeros((n_obs,n_ens))
    if(n_obs_sat>0): HX[0:n_obs_sat,:] = np.matmul(H[0:n_obs_sat,sig_mask],Bsat)
    HX[n_obs_sat:,:] = np.matmul(H[n_obs_sat:,:],Xtmp)
    
    ### Compute derivatives of Bsat w.r.t. sigma
    HdBsat = np.zeros((n_obs_sat,n_ens))
    dB1_dsig = -k*eta1**(k-1)
    dB2_dsig = k*eta2**(k-1)*(1/dsig_detab(eta2,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale))
    dalpha1_dsig = 95*(np.exp(-(-95*X[sig_mask,:]+21.5)**2))/np.sqrt(np.pi)
    dalpha2_dsig = -0.425*2*95*(np.exp(-(-95*X[sig_mask,:]+21.5)**2))/np.sqrt(np.pi)
    dalpha3_dsig = -5*(np.exp(-(-5*X[sig_mask,:]+3)**2))/np.sqrt(np.pi)
    dalpha4_dsig = -3*(np.exp(-(-3*X[sig_mask,:]-1.16)**2))/np.sqrt(np.pi)
    dBsat_dsig = alpha1*alpha3*dB1_dsig + alpha1*dalpha3_dsig*B1 + B1*dalpha1_dsig*alpha3 + (alpha2+alpha4)*dB2_dsig + B2*(dalpha2_dsig+dalpha4_dsig)
    if(n_obs_sat>0): HdBsat = np.matmul(H[0:n_obs_sat,sig_mask],dBsat_dsig)

    return HX, HdBsat

#############################################################################
# Jacobian matrix for satellite observation

def jacobian(HdBsat_i,H,T,Nk_fc,k,dres,sat_vel,n_obs_sat,sat_init_pos):

    sat_pos = ((sat_init_pos*Nk_fc+sat_vel*Nk_fc*(T+1))%Nk_fc).astype(int)
    J = np.copy(H)
    J[0:n_obs_sat,sat_pos] = HdBsat_i

    return J

##############################################################################
### Modulated ensemble: Bishop et al. (2017) + Houtekamer & Mitchell
### This allows us to use a fully non-linear observation operator keeping model-space localisation

def Gain_modens_HM(rho,K,n_d,Xbar,Xdev,sig_mask,k,H,sig_c,sig_r,n_obs,n_obs_sat,R,h5_file_data):
    
    import scipy.linalg as sl
    import scipy.special as sp   
    from f_enkf_ismodRSW import gaspcohn_sqrt_matrix

    W = gaspcohn_sqrt_matrix(rho,n_d)
    L = np.size(W,axis=1)
    M = L*K
    Z = np.zeros((n_d,M))
    for j in range(L):
        Z[:,(j*K):((j+1)*K)] = (1./np.sqrt(K-1)) * W[:,j,None] * Xdev
    v = np.array([Xbar + np.sqrt(M)*Z[:,k] for k in range(M)]).T
    vbar = np.repeat(np.mean(v,axis=1),M).reshape(n_d,M)
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
    v[sig_mask,:] = vB2 # this is done for when temperature is observed

    ### Compute the projection of the model onto the observation space
    Hv = np.zeros((n_obs,M)) 
    if(n_obs_sat>0): 
        Hv[0:n_obs_sat,:] = np.matmul(H[0:n_obs_sat,sig_mask],vBsat)
    Hv[n_obs_sat:,:] = np.matmul(H[n_obs_sat:,:],v)
       
    Hvbar = np.repeat(np.mean(Hv,axis=1),M).reshape(n_obs, M)
    Hvdev = Hv - Hvbar
    Pf_loc = np.matmul(vdev,vdev.T)/M
    Pf = rho*np.matmul(Xdev,Xdev.T)/(K-1)
    PfHT = np.matmul(vdev,Hvdev.T)/M
    HPfHT = np.matmul(Hvdev,Hvdev.T)/M
    Ktemp = HPfHT + R
    Ktemp = np.linalg.inv(Ktemp)
    K = np.matmul(PfHT, Ktemp) 
    
    return K

##############################################################################
### Function to compute the derivative of Bsat for the OID

def dHdxK(Xan_i,sig_mask,sig_r,sig_c,n_obs,H,K,n_obs_sat,n_obs_T,R,k,theta1,theta2,eta0,g,Z0,U_scale,h5_file_data):
    
    import scipy.special as sp   
 
    ### Manipulating Xan[sig_mask] to get radiance from each layer
    eta2 = interp_sig2etab(Xan_i[sig_mask],h5_file_data)
    eta1 = interp_sig2etab(Xan_i[sig_mask],h5_file_data) - Xan_i[sig_mask]
    B2 = eta2**k
    B1 = eta1**k

    ### Alpha weights
    alpha1 = 0.5-0.5*sp.erf(-95*Xan_i[sig_mask]+21.5)
    alpha2 = 0.425+0.425*sp.erf(-95*Xan_i[sig_mask]+21.5)
    alpha3 = 0.5+0.5*sp.erf(-5*Xan_i[sig_mask]+3)
    alpha4 = 0.5+0.5*sp.erf(-3*Xan_i[sig_mask]-1.16)

    Htmp = np.copy(H)

    ### Compute derivatives of Bsat w.r.t. sigma 
    dB1_dsig = -k*eta1**(k-1)
    dB2_dsig = k*eta2**(k-1)*(1/dsig_detab(eta2,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale))
    dalpha1_dsig = 95*(np.exp(-(-95*Xan_i[sig_mask]+21.5)**2))/np.sqrt(np.pi)
    dalpha2_dsig = -0.425*2*95*(np.exp(-(-95*Xan_i[sig_mask]+21.5)**2))/np.sqrt(np.pi)
    dalpha3_dsig = -5*(np.exp(-(-5*Xan_i[sig_mask]+3)**2))/np.sqrt(np.pi)
    dalpha4_dsig = -3*(np.exp(-(-3*Xan_i[sig_mask]-1.16)**2))/np.sqrt(np.pi)
    dBsat_dsig = alpha1*alpha3*dB1_dsig + alpha1*dalpha3_dsig*B1 + B1*dalpha1_dsig*alpha3 + (alpha2+alpha4)*dB2_dsig + B2*(dalpha2_dsig+dalpha4_dsig)
    if(n_obs_sat>0): Htmp[:n_obs_sat,sig_mask] = H[:n_obs_sat,sig_mask]*dBsat_dsig
    if(n_obs_T>0): Htmp[n_obs_sat:(n_obs_sat+n_obs_T),sig_mask] = H[n_obs_sat:(n_obs_sat+n_obs_T),sig_mask]*dB2_dsig

    HK = np.matmul(Htmp,K)

    return HK
