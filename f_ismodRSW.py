#######################################################################
# FUNCTIONS REQUIRED FOR NUMERICAL INTEGRATION OF THE ismodRSW MODEL
#######################################################################

''' 

Module contains numerous functions for the numerical inegration of the modRSW model:
> make_grid            : makes mesh for given length and gridcell number
> NCPflux4d             : calculates numerical flux as per the theory of Kent et al. 2017
> NCPflux_topog         : calcualtes flux for flow over topography
> time_step            : calculates stable time step for integration
> step_forward_modRSW   : integrates forward one time step (forward euler) using NCPflux4d 
> heaviside             : vector-aware implementation of heaviside (also works for scalars).
> ens_forecast_topog    : for use in parallel ensemble forecasting
'''
    
import math as m
import numpy as np
from isen_func import interp_sig2etab, dMdsig, M_int

##################################################################
#'''-------------- Create mesh at given resolution --------------'''
##################################################################  

# domain [0,L]
def make_grid(Nk,L):
    Kk = L/Nk                   # length of cell
    x = np.linspace(0, L, Nk+1)         # edge coordinates
    xc = np.linspace(Kk/2,L-Kk/2,Nk) # cell center coordinates
    grid = [Kk, x, xc]
    return grid

# domain [-L/2,L/2]:
def make_grid_2(Nk,L):
    Kk = L/Nk                   # length of cell
    x = np.linspace(-0.5*L, 0.5*L, Nk+1)        # edge coordinates
    xc = x - 0.5*Kk # cell center coordinates
    xc = xc[1:]
    grid = [Kk, x, xc]
    return grid

##################################################################
#'''----------------- NCP flux function -----------------'''
##################################################################        
#@jit
#def NCPflux4d(UL,UR,sig_r,sig_c,c2,beta,ML,MR,Mc,dMdsigL,dMdsigR):
def NCPflux4d(sigL,sigR,uL,uR,vL,vR,rL,rR,sig_r,sig_c,c2,beta,SL,SR,ML,MR,Mc,dMdsigL,dMdsigR,a,b,h1,h2,h3,h4,h5):

### INPUT ARGS:
# UL: left state 4-vector       e.g.: UL = np.array([1,2,0,1]) 
# UR: right state 4-vector      e.g.: UR = np.array([1.1,1.5,0,0.9])
# sig_r: threshold height r        e.g.: Hr = 1.15 (note that Hc < Hr)
# sig_c: threshold height c        e.g.: Hc = 1.02
# c2: constant for rain geop.   e.g.: c2 = 9.8*Hr/400
# beta: constant for hr eq.     e.g.: beta = 1
# ML: left state of M function
# MR: right state of M function
# Mc: M function evaluated in sig_c
# dMdsigL: left state of dM/dsig
# dMdsigR: right state of dM/dsig

### OUTPUT:
# Flux: 4-vector of flux values between left and right states
# VNC : 4-vector of VNC values due to NCPs

    UL = np.array([sigL, sigL*uL, sigL*vL, sigL*rL])
    UR = np.array([sigR, sigR*uR, sigR*vR, sigR*rR])

    #    '''%%%----- compute the integrals as per the theory -----%%%'''
    if a == 0:
        Ibeta = beta*h1*np.heaviside(b,1.)
        Itaubeta = 0.5*beta*h1*np.heaviside(b,1.)
    else:    
        d = (a+b)/a
        e = (np.square(a) - np.square(b))/np.square(a)
        Ibeta = beta*h1*(d*np.heaviside(a+b,1.) - (b/a)*np.heaviside(b,1.))
        Itaubeta = 0.5*beta*h1*(e*np.heaviside(a+b,1.) + (np.square(b)/np.square(a))*np.heaviside(b,1.))

    #    '''%%%----- VNC component -----%%%'''
    VNC1 = 0.
    VNC2 = -c2*(rL-rR)*0.5*(sigL+sigR)
    VNC3 = 0.
    VNC4 = -h1*(uL-uR)*(sigR*Ibeta - (sigL-sigR)*Itaubeta)

    VNC = np.array([VNC1, VNC2, VNC3, VNC4])
    
    #   '''%%%----- Vector flux P^NC in the theory -----%%%'''
    if SL >= 0:
        PhL = ML + (Mc - ML)*np.heaviside(sigL-sig_c,1.) 
        FluxL = np.array([sigL*uL, sigL*np.square(uL) + PhL, sigL*uL*vL, sigL*uL*rL])
        Flux = FluxL - 0.5*VNC
    elif SR <= 0:
        PhR = MR + (Mc - MR)*np.heaviside(sigR-sig_c,1.)
        FluxR = np.array([sigR*uR, sigR*np.square(uR) + PhR, sigR*uR*vR, sigR*uR*rR])
        Flux = FluxR + 0.5*VNC
    elif SL < 0 and SR > 0:
        PhL = ML + (Mc - ML)*np.heaviside(sigL-sig_c,1.)
        PhR = MR + (Mc - MR)*np.heaviside(sigR-sig_c,1.)
        FluxR = np.array([sigR*uR, sigR*np.square(uR) + PhR, sigR*uR*vR, sigR*uR*rR])
        FluxL = np.array([sigL*uL, sigL*np.square(uL) + PhL, sigL*uL*vL, sigL*uL*rL])
        FluxHLL = (FluxL*SR - FluxR*SL + SL*SR*(UR - UL))/(SR-SL)
        Flux = FluxHLL - (0.5*(SL+SR)/(SR-SL))*VNC
 
    return Flux, VNC

##################################################################
#'''----------------- Heaviside step function -----------------'''
##################################################################
def heaviside(x):
    """
    Vector-aware implemenation of the Heaviside step function.
    """
    return int(x > 0)

##################################################################
#'''--------- Compute stable timestep ---------'''
##################################################################
#@jit
def time_step(U,Kk,cfl,dM_dsig,cc2,beta):
### INPUT ARGS:
# U: array of variarible values at t
# Kk: grid size

### OUTPUT:
# dt: stable timestep (h>0 only)

    # signal velocties (calculated from eigenvalues)
    lam1 = abs(U[1,:]/U[0,:] - np.sqrt(cc2*beta + dM_dsig))
    lam2 = abs(U[1,:]/U[0,:] + np.sqrt(cc2*beta + dM_dsig))
    denom = np.maximum(lam1,lam2)
    dt = cfl*min(Kk/denom)

    return dt

##################################################################
# RANDOM CONVECTION INITIATION (as per Wursch and Craig)
##################################################################

def conv_pert_per(Nk,Kk,x,xn,U_ampl,l): # generation of random perturbations to u as per Wursch and Craig to initiate convection (in a periodic fashion)

    Fn = np.zeros(Nk)
    xp = np.linspace(-0.5,0.5,200)
    yp = np.roll(xp,int((xn-0.5)//Kk)) # to deal with perturbation in a periodic domain
    Fn = U_ampl*(-2*yp/l**2)*np.exp(-yp**2/l**2)

    return Fn

def conv_pert_dens(Nk,x,xn,sig_ampl,l):

    Fn = np.zeros(Nk)
    Fn = sig_ampl*np.exp(-(xn-x)**2/l**2)

    return Fn

def conv_pert(Nk,Kk,x,xn,U_ampl,l): # generation of random perturbations to u as per Wursch and Craig to initiate convection

    Fn = np.zeros(Nk)
    Fn = U_ampl*(-2*(x-xn)/l**2)*np.exp(-(x-xn)**2/l**2)

    return Fn

##################################################################
# ZERO TOPOGRAPHY: integrate forward one time step ...
#################################################################
    
def L_periodic(U,dt,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro):
    S = np.empty((Neq,Nk))
    S[0,:] = 0
    S[1,:] = (1./Ro)*U[2,:]
    S[2,:] = -(1./Ro)*U[1,:]
    S[3,:] = -alpha2*U[3,:]    
 
    left = list(range(0,Nk))
    right = np.append(list(range(1,Nk)),0)

    sigL = U[0,left] 
    sigL = np.append(sigL[-1],sigL)
    sigR = U[0,right]
    sigR = np.append(sigR[-1],sigR)
    uL = U[1,left]/U[0,left]
    uL = np.append(uL[-1],uL)
    uR = U[1,right]/U[0,right]
    uR = np.append(uR[-1],uR)
    vL = U[2,left]/U[0,left]
    vL = np.append(vL[-1],vL)
    vR = U[2,right]/U[0,right]
    vR = np.append(vR[-1],vR)
    rL = U[3,left]/U[0,left]
    rL = np.append(rL[-1],rL)
    rR = U[3,right]/U[0,right]
    rR = np.append(rR[-1],rR)

    ML = M[left]
    ML = np.append(ML[-1],ML)
    MR = M[right]
    MR = np.append(MR[-1],MR)

    dM_dsigL = dM_dsig[left]
    dM_dsigL = np.append(dM_dsigL[-1],dM_dsigL)
    dM_dsigR = dM_dsig[right]
    dM_dsigR = np.append(dM_dsigR[-1],dM_dsigR)
    
    h1 = np.heaviside(uL-uR,1.)
    h2 = np.heaviside(sigL-sig_r,1.)
    h3 = np.heaviside(sigR-sig_r,1.)
    h4 = np.heaviside(sig_c-sigL,1.)
    h5 = np.heaviside(sig_c-sigR,1.)

    #    '''wave speeds'''
    SL = np.minimum(uL - np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR - np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5))
    SR = np.maximum(uL + np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR + np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5)) 

    #    '''%%%----- For calculating NCP components... -----%%%'''
    a = sigR-sigL
    b = sigL-sig_r

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
   
    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(sigL[j],sigR[j],uL[j],uR[j],vL[j],vR[j],rL[j],rR[j],sig_r,sig_c,cc2,beta,SL[j],SR[j],ML[j],MR[j],Mc,dM_dsigL[j],dM_dsigR[j],a[j],b[j],h1[j],h2[j],h3[j],h4[j],h5[j])
 
    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux
    
    return -(Pp[:,1:] - Pm[:,:-1])/Kk + S

def L_inflow(U,dt,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl):

    #'''%%%----- compute extraneous forcing terms S(U) -----%%%'''
    S = np.empty((Neq,Nk))
    S[0,:] = 0
    S[1,:] = (1./Ro)*U[2,:]
    S[2,:] = -(1./Ro)*U[1,:]
    S[3,:] = -alpha2*U[3,:]  

    #'''%%%----- create a copy of U which includes ghost cells -----%%%'''
    U_ext = np.zeros((Neq,Nk+2))
    U_ext[:,1:-1] = U
    U_ext[:3,0] = U_infl # specified inflow for sig,sigu,sigv
    U_ext[:,-1] = U_ext[:,-2] # outflow condition

    M_ext  = np.empty(Nk+2)
    M_ext[1:-1] = M
    M_ext[0] = M_infl
    M_ext[-1] = M_ext[-2]

    dM_dsig_ext = np.empty(Nk+2)
    dM_dsig_ext[1:-1] = dM_dsig
    dM_dsig_ext[0] = dM_dsig_infl
    dM_dsig_ext[-1] = dM_dsig_ext[-2]
 
    left = list(range(0,Nk+1))
    right = list(range(1,Nk+2))

    sigL = U_ext[0,left] 
    sigR = U_ext[0,right]
    uL = U_ext[1,left]/U_ext[0,left]
    uR = U_ext[1,right]/U_ext[0,right]
    vL = U_ext[2,left]/U_ext[0,left]
    vR = U_ext[2,right]/U_ext[0,right]
    rL = U_ext[3,left]/U_ext[0,left]
    rR = U_ext[3,right]/U_ext[0,right]

    ML = M_ext[left]
    MR = M_ext[right]

    dM_dsigL = dM_dsig_ext[left]
    dM_dsigR = dM_dsig_ext[right]
    
    h1 = np.heaviside(uL-uR,1.)
    h2 = np.heaviside(sigL-sig_r,1.)
    h3 = np.heaviside(sigR-sig_r,1.)
    h4 = np.heaviside(sig_c-sigL,1.)
    h5 = np.heaviside(sig_c-sigR,1.)

    #    '''wave speeds'''
    SL = np.minimum(uL - np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR - np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5))
    SR = np.maximum(uL + np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR + np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5)) 

    #    '''%%%----- For calculating NCP components... -----%%%'''
    a = sigR-sigL
    b = sigL-sig_r

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
   
    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(sigL[j],sigR[j],uL[j],uR[j],vL[j],vR[j],rL[j],rR[j],sig_r,sig_c,cc2,beta,SL[j],SR[j],ML[j],MR[j],Mc,dM_dsigL[j],dM_dsigR[j],a[j],b[j],h1[j],h2[j],h3[j],h4[j],h5[j])
 
    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux

    return -(Pp[:,1:] - Pm[:,:-1])/Kk + S

def L_inflow_outflow(U,dt,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl,U_oufl,M_oufl,dM_dsig_oufl):

    #'''%%%----- compute extraneous forcing terms S(U) -----%%%'''
    S = np.empty((Neq,Nk))
    S[0,:] = 0
    S[1,:] = (1./Ro)*U[2,:]
    S[2,:] = -(1./Ro)*U[1,:]
    S[3,:] = -alpha2*U[3,:]  

    #'''%%%----- create a copy of U which includes ghost cells -----%%%'''
    U_ext = np.zeros((Neq,Nk+2))
    U_ext[:,1:-1] = U
    U_ext[:3,0] = U_infl # specified inflow for sig,sigu,sigv
    U_ext[:3,-1] = U_oufl # specified outflow 

    M_ext  = np.empty(Nk+2)
    M_ext[1:-1] = M
    M_ext[0] = M_infl
    M_ext[-1] = M_oufl

    dM_dsig_ext = np.empty(Nk+2)
    dM_dsig_ext[1:-1] = dM_dsig
    dM_dsig_ext[0] = dM_dsig_infl
    dM_dsig_ext[-1] = dM_dsig_oufl
 
    left = list(range(0,Nk+1))
    right = list(range(1,Nk+2))

    sigL = U_ext[0,left] 
    sigR = U_ext[0,right]
    uL = U_ext[1,left]/U_ext[0,left]
    uR = U_ext[1,right]/U_ext[0,right]
    vL = U_ext[2,left]/U_ext[0,left]
    vR = U_ext[2,right]/U_ext[0,right]
    rL = U_ext[3,left]/U_ext[0,left]
    rR = U_ext[3,right]/U_ext[0,right]

    ML = M_ext[left]
    MR = M_ext[right]

    dM_dsigL = dM_dsig_ext[left]
    dM_dsigR = dM_dsig_ext[right]
    
    h1 = np.heaviside(uL-uR,1.)
    h2 = np.heaviside(sigL-sig_r,1.)
    h3 = np.heaviside(sigR-sig_r,1.)
    h4 = np.heaviside(sig_c-sigL,1.)
    h5 = np.heaviside(sig_c-sigR,1.)

    #    '''wave speeds'''
    SL = np.minimum(uL - np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR - np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5))
    SR = np.maximum(uL + np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR + np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5)) 

    #    '''%%%----- For calculating NCP components... -----%%%'''
    a = sigR-sigL
    b = sigL-sig_r

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
   
    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(sigL[j],sigR[j],uL[j],uR[j],vL[j],vR[j],rL[j],rR[j],sig_r,sig_c,cc2,beta,SL[j],SR[j],ML[j],MR[j],Mc,dM_dsigL[j],dM_dsigR[j],a[j],b[j],h1[j],h2[j],h3[j],h4[j],h5[j])
 
    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux

    return -(Pp[:,1:] - Pm[:,:-1])/Kk + S


def step_forward_RK3(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro,w1,U_infl,M_infl,dM_dsig_infl):

    a10 = 1.
    a20 = 0.75
    a21 = 0.25
    a30 = 1./3.
    a31 = 0.
    a32 = 2./3.
    b10 = 1.
    b20 = 0.
    b21 = 0.25
    b30 = 0. 
    b31 = 0.
    b32 = 2./3.

    L0 = L_inflow(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl)
    U1 = a10*U0 + b10*dt*L0
    etab1 = interp_sig2etab(U1[0,:])
    dM_dsig1 = dMdsig(etab1,U1[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M1 = M_int(etab1,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale) 
    L1 = L_inflow(U1,dt,Nk,Neq,Kk,M1,Mc,sig_r,sig_c,dM_dsig1,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl)
    U2 = a20*U0 + a21*U1 + b20*dt*L0 + b21*dt*L1
    etab2 = interp_sig2etab(U2[0,:])
    dM_dsig2 = dMdsig(etab2,U2[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M2 = M_int(etab2,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    L2 = L_inflow(U2,dt,Nk,Neq,Kk,M2,Mc,sig_r,sig_c,dM_dsig2,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl)
    U3 = a30*U0 + a31*U1 + a32*U2 + b30*dt*L0 + b31*dt*L1 + b32*dt*L2

    return U3    

def step_forward_RK3_oufl(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro,w1,U_infl,M_infl,dM_dsig_infl,U_oufl,M_oufl,dM_dsig_oufl):

    a10 = 1.
    a20 = 0.75
    a21 = 0.25
    a30 = 1./3.
    a31 = 0.
    a32 = 2./3.
    b10 = 1.
    b20 = 0.
    b21 = 0.25
    b30 = 0. 
    b31 = 0.
    b32 = 2./3.

    L0 = L_inflow_outflow(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl,U_oufl,M_oufl,dM_dsig_oufl)
    U1 = a10*U0 + b10*dt*L0
    etab1 = interp_sig2etab(U1[0,:])
    dM_dsig1 = dMdsig(etab1,U1[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M1 = M_int(etab1,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale) 
    L1 = L_inflow_outflow(U1,dt,Nk,Neq,Kk,M1,Mc,sig_r,sig_c,dM_dsig1,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl,U_oufl,M_oufl,dM_dsig_oufl)
    U2 = a20*U0 + a21*U1 + b20*dt*L0 + b21*dt*L1
    etab2 = interp_sig2etab(U2[0,:])
    dM_dsig2 = dMdsig(etab2,U2[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M2 = M_int(etab2,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    L2 = L_inflow_outflow(U2,dt,Nk,Neq,Kk,M2,Mc,sig_r,sig_c,dM_dsig2,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl,U_oufl,M_oufl,dM_dsig_oufl)
    U3 = a30*U0 + a31*U1 + a32*U2 + b30*dt*L0 + b31*dt*L1 + b32*dt*L2

    return U3    

def step_forward_periodic_RK3(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro):
    
    a10 = 1.
    a20 = 0.75
    a21 = 0.25
    a30 = 1./3.
    a31 = 0.
    a32 = 2./3.
    b10 = 1.
    b20 = 0.
    b21 = 0.25
    b30 = 0.
    b31 = 0.
    b32 = 2./3.

    L0 = L_periodic(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,Ro)
    U1 = a10*U0 + b10*dt*L0
    etab1 = interp_sig2etab(U1[0,:])
    dM_dsig1 = dMdsig(etab1,U1[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M1 = M_int(etab1,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    L1 = L_periodic(U1,dt,Nk,Neq,Kk,M1,Mc,sig_r,sig_c,dM_dsig1,cc2,alpha2,beta,Ro)
    U2 = a20*U0 + a21*U1 + b20*dt*L0 + b21*dt*L1
    etab2 = interp_sig2etab(U2[0,:])
    dM_dsig2 = dMdsig(etab2,U2[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M2 = M_int(etab2,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    L2 = L_periodic(U2,dt,Nk,Neq,Kk,M2,Mc,sig_r,sig_c,dM_dsig2,cc2,alpha2,beta,Ro)
    U3 = a30*U0 + a31*U1 + a32*U2 + b30*dt*L0 + b31*dt*L1 + b32*dt*L2

    return U3

def step_forward_periodic_RK2(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro):
    
    a10 = 1.
    a20 = 0.5
    a21 = 0.5
    b10 = 1.
    b20 = 0.
    b21 = 0.5

    L0 = L_periodic(U0,dt,Nk,Neq,Kk,M0,Mc,sig_r,sig_c,dM_dsig0,cc2,alpha2,beta,Ro)
    U1 = a10*U0 + b10*dt*L0
    etab1 = interp_sig2etab(U1[0,:])
    dM_dsig1 = dMdsig(etab1,U1[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M1 = M_int(etab1,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    L1 = L_periodic(U1,dt,Nk,Neq,Kk,M1,Mc,sig_r,sig_c,dM_dsig1,cc2,alpha2,beta,Ro)
    U2 = a20*U0 + a21*U1 + b20*dt*L0 + b21*dt*L1

    return U2

#@jit
def step_forward_ismodRSW(U,U_rel,dt,tn,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,tau_rel):
### INPUT ARGS:
# U: array of variarible values at t, size (Neq,Nk)
# dt: stable time step
# Nk, Kk: mesh info


### OUTPUT:
# UU: array of variarible values at t+1, size (Neq,Nk)

    #'''%%%----- compute extraneous forcing terms S(U) -----%%%'''
    S = np.empty((Neq,Nk))
    S[0,:] = 0
    S[1,:] = (1./Ro)*U[2,:]
    S[2,:] = -(1./Ro)*U[1,:] + (U[0,:]*U_rel[2,:]-U[2,:])/tau_rel
    S[3,:] = -alpha2*U[3,:]    
 
    left = list(range(0,Nk))
    right = np.append(list(range(1,Nk)),0)

    sigL = U[0,left] 
    sigL = np.append(sigL[-1],sigL)
    sigR = U[0,right]
    sigR = np.append(sigR[-1],sigR)
    uL = U[1,left]/U[0,left]
    uL = np.append(uL[-1],uL)
    uR = U[1,right]/U[0,right]
    uR = np.append(uR[-1],uR)
    vL = U[2,left]/U[0,left]
    vL = np.append(vL[-1],vL)
    vR = U[2,right]/U[0,right]
    vR = np.append(vR[-1],vR)
    rL = U[3,left]/U[0,left]
    rL = np.append(rL[-1],rL)
    rR = U[3,right]/U[0,right]
    rR = np.append(rR[-1],rR)

    ML = M[left]
    ML = np.append(ML[-1],ML)
    MR = M[right]
    MR = np.append(MR[-1],MR)

    dM_dsigL = dM_dsig[left]
    dM_dsigL = np.append(dM_dsigL[-1],dM_dsigL)
    dM_dsigR = dM_dsig[right]
    dM_dsigR = np.append(dM_dsigR[-1],dM_dsigR)
    
    h1 = np.heaviside(uL-uR,1.)
    h2 = np.heaviside(sigL-sig_r,1.)
    h3 = np.heaviside(sigR-sig_r,1.)
    h4 = np.heaviside(sig_c-sigL,1.)
    h5 = np.heaviside(sig_c-sigR,1.)

    #    '''wave speeds'''
    SL = np.minimum(uL - np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR - np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5))
    SR = np.maximum(uL + np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR + np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5)) 

    #    '''%%%----- For calculating NCP components... -----%%%'''
    a = sigR-sigL
    b = sigL-sig_r

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
   
    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(sigL[j],sigR[j],uL[j],uR[j],vL[j],vR[j],rL[j],rR[j],sig_r,sig_c,cc2,beta,SL[j],SR[j],ML[j],MR[j],Mc,dM_dsigL[j],dM_dsigR[j],a[j],b[j],h1[j],h2[j],h3[j],h4[j],h5[j])

    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux
  
    UU = np.empty(np.shape(U))
    UU = U - dt*(Pp[:,1:] - Pm[:,:-1])/Kk + dt*S
        
    return UU

##################################################################

def step_forward_ismodRSW_inflow(U,dt,tn,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,w1,U_infl,M_infl,dM_dsig_infl):
### INPUT ARGS:
# U: array of variarible values at t, size (Neq,Nk)
# dt: stable time step
# Nk, Kk: mesh info


### OUTPUT:
# UU: array of variarible values at t+1, size (Neq,Nk)

    #'''%%%----- compute extraneous forcing terms S(U) -----%%%'''
    S = np.empty((Neq,Nk))
    S[0,:] = 0
    S[1,:] = (1./Ro)*U[2,:]
    S[2,:] = -(1./Ro)*U[1,:]
    S[3,:] = -alpha2*U[3,:]  

    #'''%%%----- create a copy of U which includes ghost cells -----%%%'''
    U_ext = np.zeros((Neq,Nk+2))
    U_ext[:,1:-1] = U
    U_ext[:3,0] = U_infl # specified inflow for sig,sigu,sigv
    U_ext[:,-1] = U_ext[:,-2] # outflow condition

    M_ext  = np.empty(Nk+2)
    M_ext[1:-1] = M
    M_ext[0] = M_infl
    M_ext[-1] = M_ext[-2]

    dM_dsig_ext = np.empty(Nk+2)
    dM_dsig_ext[1:-1] = dM_dsig
    dM_dsig_ext[0] = dM_dsig_infl
    dM_dsig_ext[-1] = dM_dsig_ext[-2]
 
    left = list(range(0,Nk+1))
    right = list(range(1,Nk+2))

    sigL = U_ext[0,left] 
    sigR = U_ext[0,right]
    uL = U_ext[1,left]/U_ext[0,left]
    uR = U_ext[1,right]/U_ext[0,right]
    vL = U_ext[2,left]/U_ext[0,left]
    vR = U_ext[2,right]/U_ext[0,right]
    rL = U_ext[3,left]/U_ext[0,left]
    rR = U_ext[3,right]/U_ext[0,right]

    ML = M_ext[left]
    MR = M_ext[right]

    dM_dsigL = dM_dsig_ext[left]
    dM_dsigR = dM_dsig_ext[right]
    
    h1 = np.heaviside(uL-uR,1.)
    h2 = np.heaviside(sigL-sig_r,1.)
    h3 = np.heaviside(sigR-sig_r,1.)
    h4 = np.heaviside(sig_c-sigL,1.)
    h5 = np.heaviside(sig_c-sigR,1.)

    #    '''wave speeds'''
    SL = np.minimum(uL - np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR - np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5))
    SR = np.maximum(uL + np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR + np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5)) 

    #    '''%%%----- For calculating NCP components... -----%%%'''
    a = sigR-sigL
    b = sigL-sig_r

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
   
    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(sigL[j],sigR[j],uL[j],uR[j],vL[j],vR[j],rL[j],rR[j],sig_r,sig_c,cc2,beta,SL[j],SR[j],ML[j],MR[j],Mc,dM_dsigL[j],dM_dsigR[j],a[j],b[j],h1[j],h2[j],h3[j],h4[j],h5[j])
 
    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux
    
    #'''%%%----- step forward in time -----%%%'''

    UU = np.empty((Neq,Nk))
    UU = U - dt*(Pp[:,1:] - Pm[:,:-1])/Kk + dt*S

    return UU

def step_forward_ismodRSW_inflow_outflow(U,dt,tn,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,BC,w1,U_infl,U_oufl,M_infl,M_oufl,dM_dsig_infl,dM_dsig_oufl):
### INPUT ARGS:
# U: array of variarible values at t, size (Neq,Nk)
# dt: stable time step
# Nk, Kk: mesh info


### OUTPUT:
# UU: array of variarible values at t+1, size (Neq,Nk)

    #'''%%%----- compute extraneous forcing terms S(U) -----%%%'''
    S = np.empty((Neq,Nk))
    S[0,:] = 0
    S[1,:] = (1./Ro)*U[2,:]
    S[2,:] = -(1./Ro)*U[1,:]
    S[3,:] = -alpha2*U[3,:]  

    #'''%%%----- create a copy of U which includes ghost cells -----%%%'''
    U_ext = np.zeros((Neq,Nk+2))
    U_ext[:,1:-1] = U
    U_ext[:3,0] = U_infl # specified inflow for sig,sigu,sigv
    U_ext[:3,-1] = U_oufl # outflow condition

    M_ext  = np.empty(Nk+2)
    M_ext[1:-1] = M
    M_ext[0] = M_infl
    M_ext[-1] = M_oufl

    dM_dsig_ext = np.empty(Nk+2)
    dM_dsig_ext[1:-1] = dM_dsig
    dM_dsig_ext[0] = dM_dsig_infl
    dM_dsig_ext[-1] = dM_dsig_oufl
 
    left = list(range(0,Nk+1))
    right = list(range(1,Nk+2))

    sigL = U_ext[0,left] 
    sigR = U_ext[0,right]
    uL = U_ext[1,left]/U_ext[0,left]
    uR = U_ext[1,right]/U_ext[0,right]
    vL = U_ext[2,left]/U_ext[0,left]
    vR = U_ext[2,right]/U_ext[0,right]
    rL = U_ext[3,left]/U_ext[0,left]
    rR = U_ext[3,right]/U_ext[0,right]

    ML = M_ext[left]
    MR = M_ext[right]

    dM_dsigL = dM_dsig_ext[left]
    dM_dsigR = dM_dsig_ext[right]
    
    h1 = np.heaviside(uL-uR,1.)
    h2 = np.heaviside(sigL-sig_r,1.)
    h3 = np.heaviside(sigR-sig_r,1.)
    h4 = np.heaviside(sig_c-sigL,1.)
    h5 = np.heaviside(sig_c-sigR,1.)

    #    '''wave speeds'''
    SL = np.minimum(uL - np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR - np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5))
    SR = np.maximum(uL + np.sqrt(cc2*beta*h1*h2 + dM_dsigL*h4),uR + np.sqrt(cc2*beta*h1*h3 + dM_dsigR*h5)) 

    #    '''%%%----- For calculating NCP components... -----%%%'''
    a = sigR-sigL
    b = sigL-sig_r

    #''' %%%----- Determine intercell fluxes using numerical flux function -----%%%'''
    Flux = np.empty((Neq,Nk+1))
    VNC = np.empty((Neq,Nk+1))
   
    for j in range(0,Nk+1):
        Flux[:,j], VNC[:,j] = NCPflux4d(sigL[j],sigR[j],uL[j],uR[j],vL[j],vR[j],rL[j],rR[j],sig_r,sig_c,cc2,beta,SL[j],SR[j],ML[j],MR[j],Mc,dM_dsigL[j],dM_dsigR[j],a[j],b[j],h1[j],h2[j],h3[j],h4[j],h5[j])
 
    Pp = 0.5*VNC + Flux
    Pm = -0.5*VNC + Flux
    
    #'''%%%----- step forward in time -----%%%'''

    UU = np.empty((Neq,Nk))
    UU = U - dt*(Pp[:,1:] - Pm[:,:-1])/Kk + dt*S

    return UU

def forecast_step_ismodRSW(N, U_forec, U_rel, tau_rel, q, Neq, n_ens, n_d, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Nforec, Mc, sig_c, sig_r, cc2, beta, alpha2, Ro, R, k, theta1, theta2, eta0, g, Z0, U_scale, Q, NIAU, h5_file_data):
            
    #######################################
    #  generate a *Nforec*-long forecast # 
    #######################################

    print(' Long-range forecast starting... ')
                
    tforec = tmeasure
    tendforec = tforec + (Nforec-1)*dtmeasure
    forec_time = np.linspace(tforec,tendforec,Nforec+1)
    forec_T = 1
    U = U_forec[:,:,forec_T-1]

    while tforec < tendforec and forec_T < Nforec:
                
       if forec_T > NIAU: 
           q[:,forec_T] = 0.0
  
       tn = forec_time[forec_T-1]
    
       while tn < (tforec+dtmeasure):

           etab = interp_sig2etab(U[0,:],h5_file_data) 
           dM_dsig = dMdsig(etab,U[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
           M = M_int(etab,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
        
           dt = time_step(U,Kk_fc,cfl_fc,dM_dsig,cc2,beta) # compute stable time step
           tn = tn + dt

           if tn > (tforec+dtmeasure):
               dt = dt - (tn - (tforec+dtmeasure)) + 1e-12
               tn = tforec + dtmeasure + 1e-12
        
           # Inject a portion of the additive noise each timestep using an
           # Incremental Analysis Update approach (Bloom et al., 1996).
           U += (q[:,forec_T].reshape(Neq, Nk_fc)) * dt / dtmeasure
           # if hr < 0, set to zero:
           hr = U[3, :]
           hr[hr < 0.] = 0.
           U[3, :] = hr
   
           # if h < 0, set to epsilon:
           h = U[0, :]
           h[h < 0.] = 1e-3
           U[0, :] = h
        
           U = step_forward_ismodRSW(U,U_rel,dt,tn,Nk_fc,Neq,Kk_fc,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,tau_rel)

       U_forec[:,:,forec_T] = U
                
       print('Ensemble member ',N,' integrated forward from time =', round(tforec,3) ,' to', round(tforec+dtmeasure,3))
       
       tforec = tforec+dtmeasure
       forec_T = forec_T + 1

    return U_forec

def ens_forecast(N, U, U_rel, tau_rel, q, Neq, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Mc, sig_c, sig_r, cc2, beta, alpha2, Ro, R, k, theta1, theta2, eta0, g, Z0, U_scale, h5_file_data):
    
    tn = assim_time[index]   
 
    while tn < tmeasure:

        etab = interp_sig2etab(U[0,:,N],h5_file_data) 
        dM_dsig = dMdsig(etab,U[0,:,N],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
        M = M_int(etab,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
        
        dt = time_step(U[:,:,N],Kk_fc,cfl_fc,dM_dsig,cc2,beta) # compute stable time step
        tn = tn + dt

        if tn > tmeasure:
            dt = dt - (tn - tmeasure) + 1e-12
            tn = tmeasure + 1e-12
        
        # Inject a portion of the additive noise each timestep using an
        # Incremental Analysis Update approach (Bloom et al., 1996).
        U[:,:,N] += (q[:,N].reshape(Neq, Nk_fc)) * dt / dtmeasure
        # if hr < 0, set to zero:
        hr = U[3,:,N]
        hr[hr < 0.] = 0.
        U[3,:,N] = hr
   
        # if h < 0, set to epsilon:
        h = U[0,:,N]
        h[h < 0.] = 1e-3
        U[0,:,N] = h
        
        U[:,:,N] = step_forward_ismodRSW(U[:,:,N],U_rel,dt,tn,Nk_fc,Neq,Kk_fc,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,tau_rel)

    return U[:,:,N]

##################################################################
