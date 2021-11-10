##################################################################
#----------------- Initial conditions for modRSW -----------------
##################################################################

'''
Functions generate different initial conditions described below for modRSW model with
and without bottom topography...

INPUT ARGS:
# x: mesh coords
# Neq: number of equations (variables) - 4 w/o topography, 5 w/ topography
# Nk: no. of cells in mesh
# H0: reference (scaled) height 
# L: length of domain
# A: amplitude
# V: velocity scale

OUTPUT:
# U0: array of initial data, size (Neq,Nk)

##################################################################
DESCRIPTIONS:

Rotation, no topography:

<init_cond_1>
--- sinusiodal waves in h and u, zero v and r.

<init_cond_2>
--- Rossby adj with disturbed height profile:
--- Exact step in h, zero u, v, and r.

<init_cond_3>
--- Rossby adj with disturbed height profile:
--- Smoothed step in h, zero u, v, and r.

<init_cond_4>
--- Rossby adj with disturbed v-velocity profile:
--- Single jet in v, flat h profile, zero u and r.

<init_cond_5>
--- Rossby adj with disturbed v-velocity profile:
--- Double jet in v, flat h profile, zero u and r.

<init_cond_6>
--- Rossby adj with disturbed v-velocity profile:
--- Quadrupel jet in v, flat h profile, zero u and r.

<init_cond_6_1>
--- Rossby adj with disturbed v-velocity profile:
--- Quadrupel jet in v, flat h=1 profile, u = constant \ne 0, and zero r.

Topography, no rotation:

<init_cond_topog>
--- single parabolic ridge

<init_cond_topog4>
--- 4 parabolic ridges

<init_cond_topog_cos>
--- superposition of sinusoids, as used in thesis chapter 6
'''

###############################################################

import numpy as np
 
###############################################################

def init_cond_rest(x,Nk,Neq,S0,L,A,V):

    ic1 = S0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

    # Define array and fill with first-order FV (piecewise constant) initial data  
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr
   
    B = np.zeros(Nk)

    return U0,B

def init_cond_1(x,Nk,Neq,H0,L,A,V):

    k = 2*np.pi # for sinusoidal waves

    ic1 = H0 + A*np.sin(2*k*x/L)
    ic2 = A*np.sin(1*k*x/L)
    #ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

    # Define array and fill with first-order FV (piecewise constant) initial data  
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv

    B = np.zeros(Nk)

    return U0, B

###############################################################

def init_cond_2(x,Nk,Neq,S0,L,A,V):
# for disturbed height (top-hat) Rossby adj. set up.
# Exact step:
    f1 = np.heaviside(x+0.25*L,1.)
    f2 = np.heaviside(x-0.25*L,1.)
    f3 = np.heaviside(x+0.75*L,1.)
    f4 = np.heaviside(x-0.75*L,1.)

    ic1 = S0 + A*(0.5*f2 - 0.5*f1 +0.5*f3 - 0.5*f4)
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sigma
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigv

    B = np.zeros(Nk)

    return U0, B

###############################################################

def init_cond_3(x,Nk,Neq,S0,L,A,V):

# for disturbed height (top-hat) Rossby adj. set up
# Smoothed step:
    gam = 15
    f1 = 1-np.tanh(gam*(x-0.6*L))
    f2 = 1-np.tanh(gam*(x-0.4*L))

    ic1 = S0 + A*(0.5*f1 - 0.5*f2)
   #ic1 = S0*np.ones(len(x))
   #ic2 = 0.1*np.ones(len(x))
   #ic2 = -0.25 + 0.5*(0.5*f1 - 0.5*f2)
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr

    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_4(x,Nk,Neq,S0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = S0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic4 = np.zeros(len(x))
# single jet
    Lj = 0.1*L
    ic3 = np.ones(len(x))
    ic3 = V*(1+np.tanh(4*(x-0.5*L)/Lj + 2))*(1-np.tanh(4*(x-0.5*L)/Lj - 2))/4 
#    ic3 = V*(1+np.tanh(4*(x)/Lj + 2))*(1-np.tanh(4*(x)/Lj - 2))/4 
    
# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr
    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_5(x,Nk,Neq,S0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = S0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic4 = np.zeros(len(x))
 
## double jet
    Lj = 0.1*L
    f1 = V*(1+np.tanh(4*(x-0.75*L)/Lj + 2))*(1-np.tanh(4*(x-0.75*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(x-0.25*L)/Lj + 2))*(1-np.tanh(4*(x-0.25*L)/Lj - 2))/4
    ic3 = f1-f2
    
# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr
    B = np.zeros(Nk)

    return U0,B
    
###############################################################

def init_cond_5_1(x,Nk,Neq,H0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = H0*np.ones(len(x))
    ic2 = 0.5*np.ones(len(x))
    ic4 = np.zeros(len(x))
 
## double jet
    Lj = 0.1*L
    f1 = V*(1+np.tanh(4*(x-0.75*L)/Lj + 2))*(1-np.tanh(4*(x-0.75*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(x-0.25*L)/Lj + 2))*(1-np.tanh(4*(x-0.25*L)/Lj - 2))/4
    ic3 = f1-f2
    
# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # hv
    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_6(x,Nk,Neq,S0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = S0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))
## multiple (>2) jets
    Lj = 0.05
    f3 = (1+np.tanh(4*(x-0.8)/Lj + 2))*(1-np.tanh(4*(x-0.8)/Lj - 2))/4
    f4 = (1+np.tanh(4*(x-0.2)/Lj + 2))*(1-np.tanh(4*(x-0.2)/Lj - 2))/4
    f5 = (1+np.tanh(4*(x-0.6)/Lj + 2))*(1-np.tanh(4*(x-0.6)/Lj - 2))/4
    f6 = (1+np.tanh(4*(x-0.4)/Lj + 2))*(1-np.tanh(4*(x-0.4)/Lj - 2))/4
    #ic4 = V*(f3+f4-f5-f6)
    ic3 = V*(f3-f4+f5-f6)

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr
    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_6_1(x,Nk,Neq,S0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = S0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))
## multiple (>2) jets
    Lj = 0.05
    f3 = (1+np.tanh(4*(x-0.8)/Lj + 2))*(1-np.tanh(4*(x-0.8)/Lj - 2))/4
    f4 = (1+np.tanh(4*(x-0.2)/Lj + 2))*(1-np.tanh(4*(x-0.2)/Lj - 2))/4
    f5 = (1+np.tanh(4*(x-0.6)/Lj + 2))*(1-np.tanh(4*(x-0.6)/Lj - 2))/4
    f6 = (1+np.tanh(4*(x-0.4)/Lj + 2))*(1-np.tanh(4*(x-0.4)/Lj - 2))/4
    #ic4 = V*(f3+f4-f5-f6)
    ic3 = V*(f3-f4-f5+f6)

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr
    B = np.zeros(Nk)

    return U0,B

#########################################################

def init_cond_6_2(x,Nk,Neq,S0,L,A,V):
# for transverse jet Rossby adj. set-up
    ic1 = S0*np.ones(len(x))
    ic2 = np.ones(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))
## multiple (>2) jets
    Lj = 0.05
    f3 = (1+np.tanh(4*(x-0.8)/Lj + 2))*(1-np.tanh(4*(x-0.8)/Lj - 2))/4
    f4 = (1+np.tanh(4*(x-0.2)/Lj + 2))*(1-np.tanh(4*(x-0.2)/Lj - 2))/4
    f5 = (1+np.tanh(4*(x-0.6)/Lj + 2))*(1-np.tanh(4*(x-0.6)/Lj - 2))/4
    f6 = (1+np.tanh(4*(x-0.4)/Lj + 2))*(1-np.tanh(4*(x-0.4)/Lj - 2))/4
    #ic4 = V*(f3+f4-f5-f6)
    ic3 = V*(f3-f4-f5+f6)

# Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr
    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_7(x,Nk,Neq,S0,L,A,V):

    ic1 = S0*np.ones(len(x))
    ic2 = V*np.ones(len(x))
    ic3 = np.zeros(len(x))
    ic4 = np.zeros(len(x))

    # Define array and fill with first-order FV (piecewise constant) initial data  
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr
   
    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_8(x,Nk,Neq,S0,L,A,V,LD,acc,Ro):

    dsig = A
    a = 0.4

    s0 = 2*dsig*(L-2*a)/L
    k = (lambda m: 2*np.pi*m/L)
    sk = (lambda m: (1/(k(m)**2*LD**2+1))*(2*dsig*(np.sin(k(m)*(L-a))/k(m)-np.sin(k(m)*a)/k(m))/L))
    rk = (lambda m: (1/(k(m)**2*LD**2+1))*(2*dsig*(np.cos(k(m)*a)/k(m)-np.cos(k(m)*(L-a))/k(m))/L))
    S_k = 0.
    R_k = 0.
    kS_k = 0.
    kR_k = 0.

    for i in range(1,acc):
        S_k += sk(i)*np.cos(k(i)*x)
        R_k += rk(i)*np.sin(k(i)*x)
        kS_k += (-k(i))*sk(i)*np.sin(k(i)*x)
        kR_k += k(i)*rk(i)*np.cos(k(i)*x) 
 
    sig = s0/2 + S_k + R_k
    v = (LD**2/(Ro*S0))*(kS_k + kR_k)

    ic1 = np.ones(len(x))*S0# + sig
    ic2 = np.zeros(len(x))
    ic3 = v
    ic4 = np.zeros(len(x))

    # Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr

    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_9(x,Nk,Neq,S0,L,A,V): # multiple top-hats

    a1 = 0.1
    a2 = 0.1
    a3 = 0.05

    ic1 = S0+A*(np.heaviside(a1-abs(x-L/4),1.)+np.heaviside(a2-abs(x-3*L/4),1.)+np.heaviside(a3-abs(x-L/2),1.))
    ic2 = np.zeros(len(x))
    ic3 = np.ones(len(x))
    ic4 = np.zeros(len(x))
   
    Lj = 0.1*L 
    f1 = V*(1+np.tanh(4*(x-0.75*L)/Lj + 2))*(1-np.tanh(4*(x-0.75*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(x-0.25*L)/Lj + 2))*(1-np.tanh(4*(x-0.25*L)/Lj - 2))/4
    ic3 = f1-f2
    #ic3 = V*(1+np.tanh(4*(x-0.5*L)/Lj + 2))*(1-np.tanh(4*(x-0.5*L)/Lj - 2))/4 

    # Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(ic1[0:Nk]*ic4[0:Nk] + ic1[1:Nk+1]*ic4[1:Nk+1]) # sigr

    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_10(x,Nk,Neq,S0,L,A,V): # multiple top-hats

    a1 = 0.1
    a2 = 0.1
    a3 = 0.05

    ic1 = S0+A*(np.heaviside(a1-abs(x-L/4),1.)+np.heaviside(a2-abs(x-3*L/4),1.)+np.heaviside(a3-abs(x-L/2),1.))
    ic2 = np.zeros(len(x))
    ic3 = np.ones(len(x))
    ic4 = 0.1*np.sin(np.pi*x/L)
   
    Lj = 0.1*L 
    f1 = V*(1+np.tanh(4*(x-0.75*L)/Lj + 2))*(1-np.tanh(4*(x-0.75*L)/Lj - 2))/4
    f2 = V*(1+np.tanh(4*(x-0.25*L)/Lj + 2))*(1-np.tanh(4*(x-0.25*L)/Lj - 2))/4
    ic3 = f1-f2
    #ic3 = V*(1+np.tanh(4*(x-0.5*L)/Lj + 2))*(1-np.tanh(4*(x-0.5*L)/Lj - 2))/4 

    # Define array and fill with first-order FV (piecewise constant) initial data 
    U0 = np.zeros((Neq,Nk))
    U0[0,:] = 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) # sig
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # sigu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # sigv
    U0[3,:] = 0.5*(0.15*ic4[0:Nk] + 0.15*ic4[1:Nk+1]) # sigr

    B = np.zeros(Nk)

    return U0,B

###############################################################

def init_cond_topog(x,Nk,Neq,H0,L,A,V):
    # for a single parabolic ridge
    ic1 = H0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic2= 1./ic1 # for hu = 1:
    ic3 = np.zeros(len(x))
    
    # single hill
    bc = 0.5
    xp = 0.1
    a = 0.05*L
    B = np.maximum(0, bc*(1 - ((x - L*xp)**2)/a**2))
    B = np.maximum(0,B)

    U0 = np.zeros((Neq,Nk))
    B = 0.5*(B[0:Nk] + B[1:Nk+1]); # b
    U0[0,:] = np.maximum(0, 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) - B) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr

    return U0, B

###############################################################

def init_cond_topog4(x,Nk,Neq,H0,L,A,V):
    # for 4 parabolic ridges
    ic1 = H0*np.ones(len(x))
    ic2 = np.zeros(len(x))
    ic2=1/ic1 # for hu = 1:
    ic3 = np.zeros(len(x))
    
    # 4 hills
    bc = 0.4
    xp = 0.5
    a = 0.025*L
    B = np.maximum(bc*(1 - ((x - L*0.25*xp)**2)/a**2), bc*(1 - ((x - L*0.45*xp)**2)/a**2))
    B = np.maximum(B, bc*(1 - ((x - L*0.65*xp)**2)/a**2))
    B = np.maximum(B, bc*(1 - ((x - L*0.85*xp)**2)/a**2))
    B = np.maximum(0,B)
    
    U0 = np.zeros((Neq,Nk))
    B = 0.5*(B[0:Nk] + B[1:Nk+1]); # b
    U0[0,:] = np.maximum(0, 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) - B) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    
    return U0, B

###############################################################


def init_cond_topog_cos(x,Nk,Neq,sig0,H0,L,A,V):
#    superposition of cosines
    ic1 = sig0*np.ones(len(x))
    ic2= np.ones(len(x)) # for hu = 1:
    ic3 = np.zeros(len(x))

    k = 2*np.pi
    xp = 0.1
    waven = [2,4,6]
    A = H0*[100., 50., 100.]

    B = A[0]*(1+np.cos(k*(waven[0]*(x-xp)-0.5)))+ A[1]*(1+np.cos(k*(waven[1]*(x-xp)-0.5)))+    A[2]*(1+np.cos(k*(waven[2]*(x-xp)-0.5)))
    B = 0.5*B
    
    index = np.where(B<=np.min(B)+1e-10)
    index = index[0]
    B[:index[0]] = 0
    B[index[-1]:] = 0

    U0 = np.zeros((Neq,Nk))
    B = 0.5*(B[0:Nk] + B[1:Nk+1]); # b
    U0[0,:] = ic1[0:Nk] #np.maximum(0, 0.5*(ic1[0:Nk] + ic1[1:Nk+1]) - B) # h
    U0[1,:] = 0.5*(ic1[0:Nk]*ic2[0:Nk] + ic1[1:Nk+1]*ic2[1:Nk+1]) # hu
    U0[2,:] = 0.5*(ic1[0:Nk]*ic3[0:Nk] + ic1[1:Nk+1]*ic3[1:Nk+1]) # hr
    
    return U0, B

###############################################################
