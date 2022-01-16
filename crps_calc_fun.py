##################################################################
### Function to calculate the CRPS of an on ensemble of forecast variables
##################################################################
'''
    Following the theory of Hersbach (2000).
    As applied in Bowler et al (2016) and DAESR5.
    '''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
from builtins import range
import numpy as np

##################################################################

def crps_calc(X,Xa):
    '''
    Function calculates the CRPS of an on ensemble of forecast variables
    
    Input: 
    > X = X[i,:,T]: row vec with dim. N, e.g., ensemble values of a h at a given time and grid point
    > Xa = X_tr[i,0,T]: actual value of the variable at this time and gridpoint
    
    Returns crps: single value calculated as in Hersbach (2000) or DAESR5
    
    '''
    N = len(X)
    Xs = np.sort(X)
    Xs = np.insert(Xs,0,-1e12)
    Xs = np.append(Xs,1e12)


    alpha = np.zeros(N+1)
    beta = np.zeros(N+1)

    for i in range(0,N+1):
        if Xs[i+1] < Xa:
            alpha[i] = Xs[i+1] - Xs[i]
            beta[i] = 0
        elif Xs[i] > Xa:
            alpha[i] = 0
            beta[i] = Xs[i+1] - Xs[i]
        else:
            alpha[i] = Xa - Xs[i]
            beta[i] = Xs[i+1] - Xa

    alpha[0] = 0
    beta[-1] = 0

    p = np.linspace(0,1,N+1)

    crps = alpha*p**2 + beta*(1-p)**2
    crps = np.sum(crps)
    
    return crps
##################################################################
