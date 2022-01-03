#######################################################################################################################
### This script contains a collection of functions used to integrate numerically the isentropic shallow water model ###
#######################################################################################################################

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np

##################################################################
# FUNCTIONS
##################################################################

### Function to interpolate the value of the non-dimensional pressure eta using the look-up table provided in data
def interp_sig2etab(sigma,data): 
    pos = np.searchsorted(data[0,:],sigma)
    sig2 = data[0,pos]
    sig1 = data[0,pos-1]
    etab2 = data[1,pos]
    etab1 = data[1,pos-1]
    w = (sig2-sigma)/(sig2-sig1)
    etab = etab1*w + etab2*(1-w)

    return etab

###################################################################

### Function to convert the pseudo-density into non-dimensional pressure when k=1 (isopycnal limit)
def interp_sig2etab_keq1(sigma,theta1,theta2,eta0,g,R,Z0):

    etab = ((theta1-theta2)/theta1)*sigma + eta0 + (g/(R*theta1))*Z0

    return etab

###################################################################

### Function to compute the derivative of the pseudo-density with respect to the non-dimensional pressure
def dsig_detab(etab,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    dsigdetab = 1.-((1./(R*kinv*(theta1-theta2)))**kinv)*(-R*kinv*theta2*etab**(k-1.))*(-R*kinv*theta2*etab**k+R*kinv*theta1*eta0**k-g*(B-Z0))**(kinv-1.)

    return dsigdetab

###################################################################

### Function to calculate the derivative of the isentropic effective pressure with respect to the pseudo-density
def dMdsig(etab,sigma,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    dsigdetab = 1.-((1./(R*kinv*(theta1-theta2)))**kinv)*(-R*kinv*theta2*etab**(k-1.))*(-R*kinv*theta2*etab**k+R*kinv*theta1*eta0**k-g*(B-Z0))**(kinv-1.)
    dMdetab = (R*kinv*theta2/U**2)*k*sigma*etab**(k-1)   
    dMdsig = dMdetab/dsigdetab

    return dMdsig

####################################################################

### Function that computes the isentropic effective pressure
def M_int(etab,B,R,k,theta1,theta2,eta0,g,Z0,U):

    import numpy as np

    kinv = 1./k
    M = (R*kinv*theta2/U**2)*(k/(k+1.))*(etab**(k+1)+((1./((R*kinv)*(theta1-theta2)))**kinv)*(1./((R*kinv)*theta2))*((R*kinv)*theta1*eta0**k-g*(B-Z0)-(R*kinv)*theta2*etab**k)**((k+1)/k)-((1./((R*kinv)*(theta1-theta2)))**kinv)*(1./((R*kinv)*theta2))*((R*kinv)*theta1*eta0**k-g*(B-Z0))**((k+1)/k))
    
    return M

####################################################################
