import numpy as np
import h5py
import itertools as itools
import functools as ftools
from numba.decorators import jit
from numba import vectorize,float64
from builtins import range

def interp_sig2etab(sigma,data): # this function interpolate the value of etab given an array of sigma

#    h5_file = h5py.File('/home/home02/mmlca/isenRSW/table_sigma2eta.hdf','r')   
#    h5_file = h5py.File('/nobackup/mmlca/inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
#     h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
#    h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_290_theta1_300.hdf','r')
    #h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_290_theta1_300for_k_eqto_1.hdf','r')
    #h5_file = h5py.File('/home/home02/mmlca/isenRSW/sigma_eta_theta2_280_theta1_320_eta0_0.22_Z0_11000.hdf','r')
#    data = h5_file.get('sigma_eta_iversion_table')[()]
    pos = np.searchsorted(data[0,:],sigma)
    sig2 = data[0,pos]
    sig1 = data[0,pos-1]
    etab2 = data[1,pos]
    etab1 = data[1,pos-1]
    w = (sig2-sigma)/(sig2-sig1)
    etab = etab1*w + etab2*(1-w)

    return etab

def interp_sig2etab_keq1(sigma,theta1,theta2,eta0,g,R,Z0):

    etab = ((theta1-theta2)/theta1)*sigma + eta0 + (g/(R*theta1))*Z0

    return etab

#@jit
def func_sigma(B,R,k,theta1,theta2,eta0,g,Z0,U):
    
    kinv = 1./k
    sig = (lambda etab: etab-((1./(R*kinv))*(1./(theta1-theta2))*(-R*kinv*theta2*etab**k+R*kinv*theta1*eta0**k-g*(B-Z0)))**kinv)

    return sig

#@vectorize([float64(float64, float64)])
def sig2etab(sig_val,B):

#    k = R/cp
#    kinv = 1./k
#    etab = np.empty(len(sig_val))
#    for i in range(len(B)):
    etab = inversefunc(func_sigma(B),y_values=sig_val,domain=[0.8,1.2])

    return etab

#    domain_vec = np.tile(np.array([0.8,1.2]),(len(B),1))
#    etab = [inversefunc(ftools.partial(func_sigma,B[i]),sig_val[i],domain=[0.8,1.2]) for i in range(len(B))]
#    etab = map(inversefunc,map(ftools.partial,itools.repeat(func_sigma,len(sig_val)),B),sig_val,domain_vec)


def etab2sig(etab,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    sigma = etab-((1./(R*kinv))*(1./(theta1-theta2))*(-R*kinv*theta2*etab**k+R*kinv*theta1*eta0**k-g*(B-Z0)))**kinv

    return sigma

def sig2h(sigma,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    etab = interp_sig2etab(sigma)
    h = (R*kinv*theta2/g)*(etab**k-(theta2/(theta1-theta2))*(-etab**k+(theta1/theta2)*eta0**k-g*(B-Z0)/(R*kinv*theta2)))
    
    return h

def sig2hsup(sigma,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    etab = interp_sig2etab(sigma)
    h_sup = (R*kinv*theta1/g)*((1./(R*kinv))*(1./(theta1-theta2))*(-R*kinv*theta2*etab**k+R*kinv*theta1*eta0**k-g*(B-Z0))-eta0**k)

    return h_sup

def h2sig(z,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    etab = (g*(theta1-theta2)*(z-B)/(R*kinv*theta1*theta2)+eta0**k-g*(B-Z0)/(R*kinv*theta1))**kinv
    sigma = etab2sig(etab,B)
   
    return sigma

def dsig_detab(etab,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    #dsigdetab = 1 + ((theta2/(theta1-theta2))**kinv)*(((theta1/theta2)*eta0**k+(g*Z0/(R*kinv*theta2))-etab**k)**((1-k)*kinv))*(etab**(k-1))
    dsigdetab = 1.-((1./(R*kinv*(theta1-theta2)))**kinv)*(-R*kinv*theta2*etab**(k-1.))*(-R*kinv*theta2*etab**k+R*kinv*theta1*eta0**k-g*(B-Z0))**(kinv-1.)

    return dsigdetab

#@jit
def dMdsig(etab,sigma,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    dsigdetab = 1.-((1./(R*kinv*(theta1-theta2)))**kinv)*(-R*kinv*theta2*etab**(k-1.))*(-R*kinv*theta2*etab**k+R*kinv*theta1*eta0**k-g*(B-Z0))**(kinv-1.)
    dMdetab = (R*kinv*theta2/U**2)*k*sigma*etab**(k-1)   
    dMdsig = dMdetab/dsigdetab

    return dMdsig

#@jit
def M_int(etab,B,R,k,theta1,theta2,eta0,g,Z0,U):

    kinv = 1./k
    M = (R*kinv*theta2/U**2)*(k/(k+1.))*(etab**(k+1)+((1./((R*kinv)*(theta1-theta2)))**kinv)*(1./((R*kinv)*theta2))*((R*kinv)*theta1*eta0**k-g*(B-Z0)-(R*kinv)*theta2*etab**k)**((k+1)/k)-((1./((R*kinv)*(theta1-theta2)))**kinv)*(1./((R*kinv)*theta2))*((R*kinv)*theta1*eta0**k-g*(B-Z0))**((k+1)/k))
    
    return M

def montg_pot(etab,B,R,k,theta1,theta2,eta0,g,Z0,U):
    
    kinv = 1./k
    montg_pot = (R*kinv*theta2/U**2)*etab**k+(g/U**2)*B
    
    return montg_pot

def rossdef_radius(Ro,sig0,B,R,k,theta1,theta2,eta0,g,Z0,U):

    eta_0 = interp_sig2etab(sig0)
    LD = Ro*np.sqrt(sig0*dMdsig(eta_0,sig0,B,R,k,theta1,theta2,eta0,g,Z0,U))

    return LD

def c_int(sigma,B):

    k = R/cp
    kinv = 1./k
    etab_max = interp_sig2etab(sigma)
    x = np.linspace(0.946988,etab_max,100)
    #print(x)
    dM_dsig = (lambda y: ((cp*theta2/U**2)*k*((y-((1./cp)*(1./(theta1-theta2))*(-cp*theta2*y**k+cp*theta1*eta0**k-g*(B-Z0)))**kinv)*g/pref)*y**(k-1))/(1.-((1./(cp*(theta1-theta2)))**kinv)*(-cp*theta2*y**(k-1.))*(-cp*theta2*y**k+cp*theta1*eta0**k-g*(B-Z0))**(kinv-1.)))
    c = 0.
    for j in range(len(x)-1):
        c += 0.5*((np.sqrt(dM_dsig(x[j+1]))/x[j+1])*(1.-((1./(cp*(theta1-theta2)))**kinv)*(-cp*theta2*x[j+1]**(k-1.))*(-cp*theta2*x[j+1]**k+cp*theta1*eta0**k-g*(B-Z0))**(kinv-1.))+(np.sqrt(dM_dsig(x[j]))/x[j])*(1.-((1./(cp*(theta1-theta2)))**kinv)*(-cp*theta2*x[j]**(k-1.))*(-cp*theta2*x[j]**k+cp*theta1*eta0**k-g*(B-Z0))**(kinv-1.)))*(x[j+1]-x[j])
    return c
