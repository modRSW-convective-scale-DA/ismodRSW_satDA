### This script generates:
### 1) profiles of the alpha coefficients as a function of the bottom-layer pseudo-density
### 2) a profile of the (non-dimensional) brightness temperature as a function of the bottom-layer pseudo-density
### 3) a profile of the first derivative of the brightness temperature as a function of the bottom-layer pseudo-density

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import importlib
import sys
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy import interpolate
sys.path.append('..')

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

R = config.R
cp = config.cp
theta2 = config.theta2
theta1 = config.theta1
eta0 = config.eta0
Z0 = config.Z0
g = config.g
k = config.k
pref = config.pref
sigc = config.sig_c
sigr = config.sig_r

# Define additional parameter
kinv = 1./k

def dsig_deta2(eta2,B,R,k,theta1,theta2,eta0,g,Z0):

    dsigdeta2 = 1.-((1./(R*kinv*(theta1-theta2)))**kinv)*(-R*kinv*theta2*eta2**(k-1.))*(-R*kinv*theta2*eta2**k+R*kinv*theta1*eta0**k-g*(B-Z0))**(kinv-1.)

    return dsigdeta2

eta2 = np.linspace(1.011,1.111,1000)

#### Sigma2 equation
sig2 = (eta2-((1./cp)*(1./(theta1-theta2))*(-cp*theta2*eta2**k+cp*theta1*eta0**k+g*Z0))**kinv)

#### Eta1 equation
eta1 = eta2-sig2

#### Temperature definition
B1 = eta1**k
B2 = eta2**k

### Alpha weights
alpha1 = 0.5-0.5*sp.erf(-95*sig2+21.5)
alpha2 = 0.425+0.425*sp.erf(-95*sig2+21.5)
alpha3 = 0.5+0.5*sp.erf(-5*sig2+3)
alpha4 = 0.5+0.5*sp.erf(-3*sig2-1.16)


#### Brightness temperature
BT = alpha1*alpha3*B1+(alpha2+alpha4)*B2

### Derivatives
dB1_dsig = -k*eta1**(k-1)
dB2_dsig = k*eta2**(k-1)*(1/dsig_deta2(eta2,0.,R,k,theta1,theta2,eta0,g,Z0))
dalpha1_dsig = 95*(np.exp(-(-95*sig2+21.5)**2))/np.sqrt(np.pi)
dalpha2_dsig = -0.45*2*95*(np.exp(-(-95*sig2+21)**2))/np.sqrt(np.pi)
dalpha3_dsig = -5*(np.exp(-(-5*sig2+3)**2))/np.sqrt(np.pi)
dalpha4_dsig = -3*(np.exp(-(-3*sig2-1.16)**2))/np.sqrt(np.pi)
dBT_dsig = alpha1*alpha3*dB1_dsig + alpha1*dalpha3_dsig*B1 + B1*dalpha1_dsig*alpha3 + (alpha2+alpha4)*dB2_dsig + B2*(dalpha2_dsig+dalpha4_dsig)

#### Plots
fig, axes = plt.subplots(1, 3, figsize=(13,5))

### Vertical profiles of alpha coefficients
axes[0].plot(alpha1,sig2,label=u"\u03B11")
axes[0].plot(alpha2,sig2,label=u"\u03B12")
axes[0].plot(alpha3,sig2,label=u"\u03B13")
axes[0].plot(alpha4,sig2,label=u"\u03B14")
axes[0].set_xlim([0.,1.0])
axes[0].hlines(sigc,0.01,0.99,color='red',linestyle='dashed')
axes[0].hlines(sigr,0.01,0.99,color='green',linestyle='dashed')
axes[0].set_ylim([0.,0.6])
axes[0].set_xlabel('$\\alpha$',fontsize=15)
axes[0].set_ylabel('$\sigma$',fontsize=15)
axes[0].tick_params(axis='x',labelsize=15)
axes[0].tick_params(axis='y',labelsize=15)
axes[0].legend(loc=9,ncol=2,fontsize=12)

### Vertical profile of non-dimensional brightness temperature
axes[1].plot(BT,sig2,label='Brightness temp. (non-dim)')
axes[1].hlines(sigc,0.01,1.19,color='red',linestyle='dashed')
axes[1].hlines(sigr,0.01,1.19,color='green',linestyle='dashed')
axes[1].set_xlabel('B',fontsize=15)
axes[1].set_xlim([0,1.2])
axes[1].set_ylim([0.,0.6])
axes[1].tick_params(axis='x',labelsize=15)
axes[1].tick_params(axis='y',labelsize=15)
axes[1].legend(loc=9,fontsize=12)

### Vertical profile of first derivative of non-dimensional radiance
axes[2].plot(dBT_dsig,sig2,label='dB/d$\sigma$')
axes[2].plot(BT,sig2,label='B')
axes[2].set_ylim([0.,0.6])
axes[2].tick_params(axis='x',labelsize=15)
axes[2].tick_params(axis='y',labelsize=15)
axes[2].legend(loc=9, ncol=2, fontsize=12)

plt.show()
