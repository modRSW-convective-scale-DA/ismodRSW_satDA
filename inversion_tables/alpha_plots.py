import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy import interpolate

def dsig_deta2(eta2,B,R,k,theta1,theta2,eta0,g,Z0):

    kinv = 1./k
    dsigdeta2 = 1.-((1./(R*kinv*(theta1-theta2)))**kinv)*(-R*kinv*theta2*eta2**(k-1.))*(-R*kinv*theta2*eta2**k+R*kinv*theta1*eta0**k-g*(B-Z0))**(kinv-1.)

    return dsigdeta2

#### Define constants
cp = 1004.
g = 9.81
R = 287.
pref = 1000.
k = R/cp
kinv = 1./k

#### Define parameters
theta1 = 311.
theta2 = 291.8
eta0 = 0.48
Z0 = 6120.
sigc = 0.21
sigr = 0.24

eta2 = np.linspace(1.011,1.111,1000)

#### Sigma2 equation
sig2 = (eta2-((1./cp)*(1./(theta1-theta2))*(-cp*theta2*eta2**k+cp*theta1*eta0**k+g*Z0))**kinv)

#### Eta1 equation
eta1 = eta2-sig2

#### Temperature definition
B1 = eta1**k
B2 = eta2**k

### Alpha weights
#alpha1 = 0.5+0.1*np.exp(-70*((sig2-sigr)**2)/(sigr-sigc))
#erf = 0.5+0.5*sp.erf(-10*sig2+3.8)
#erf3 = 1.5+0.5*sp.erf(-10*sig2)
#alpha2 = erf*erf3*alpha1#(0.5+0.1*np.exp(-70*((sig2-sigr)**2)/(sigr-sigc)))

alpha1 = 0.5-0.5*sp.erf(-95*sig2+21.5)
alpha2 = 0.425+0.425*sp.erf(-95*sig2+21.5)
alpha3 = 0.5+0.5*sp.erf(-5*sig2+3)
alpha4 = 0.5+0.5*sp.erf(-3*sig2-1.16)


#### Brightness temperature
#BT = alpha1*B1+alpha2*B2
BT = alpha1*alpha3*B1+(alpha2+alpha4)*B2

#f1 = interpolate.interp1d(sig2,alpha1,kind='cubic')
#f2 = interpolate.interp1d(sig2,alpha2,kind='cubic')

#coeff_a1 = np.polyfit(sig2,alpha1,100)
#coeff_a2 = np.polyfit(sig2,alpha2,100)

#print(coeff_a1)
#print(coeff_a2)

#xtest = np.arange(0.,0.8,0.001)
#y1 = f1(xtest)
#y2 = f2(xtest)
#p1 = np.poly1d(coeff_a1)
#p2 = np.poly1d(coeff_a2)

### Derivatives
dB1_dsig = -k*eta1**(k-1)
dB2_dsig = k*eta2**(k-1)*(1/dsig_deta2(eta2,0.,R,k,theta1,theta2,eta0,g,Z0))
#dalpha1_dsig = 0.1*2*(-70/(sigr-sigc))*(sig2-sigr)*np.exp(-70*((sig2-sigr)**2)/(sigr-sigc))
#dalpha2_dsig = dalpha1_dsig*erf*erf3 + alpha1*(-2*0.5*10*np.exp(-(-10*sig2+3.8)**2)/np.sqrt(np.pi))*erf3+alpha1*erf*(-2*0.5*10*np.exp(-(-10*sig2)**2)/np.sqrt(np.pi))
dalpha1_dsig = 95*(np.exp(-(-95*sig2+21.5)**2))/np.sqrt(np.pi)
dalpha2_dsig = -0.45*2*95*(np.exp(-(-95*sig2+21)**2))/np.sqrt(np.pi)
dalpha3_dsig = -5*(np.exp(-(-5*sig2+3)**2))/np.sqrt(np.pi)
dalpha4_dsig = -3*(np.exp(-(-3*sig2-1.16)**2))/np.sqrt(np.pi)
#dBT_dsig = alpha1*dB1_dsig + alpha2*dB2_dsig + B1*dalpha1_dsig + B2*dalpha2_dsig
dBT_dsig = alpha1*alpha3*dB1_dsig + alpha1*dalpha3_dsig*B1 + B1*dalpha1_dsig*alpha3 + (alpha2+alpha4)*dB2_dsig + B2*(dalpha2_dsig+dalpha4_dsig)

#### Plots
fig, axes = plt.subplots(1, 3, figsize=(13,5))
axes[0].plot(alpha1,sig2,label=u"\u03B11")
axes[0].plot(alpha2,sig2,label=u"\u03B12")
axes[0].set_xlim([0.,0.8])
axes[0].hlines(sigc,0.01,0.79,color='red',linestyle='dashed')
axes[0].hlines(sigr,0.01,0.79,color='green',linestyle='dashed')
axes[0].set_ylim([0.,0.6])
axes[0].set_xlabel('$\\alpha$',fontsize=15)
axes[0].set_ylabel('$\sigma$',fontsize=15)
axes[0].tick_params(axis='x',labelsize=15)
axes[0].tick_params(axis='y',labelsize=15)
axes[0].legend(loc=9,ncol=2,fontsize=12)

axes[1].plot(BT,sig2,label='Radiance (non-dimensional)')
axes[1].hlines(sigc,0.01,1.19,color='red',linestyle='dashed')
axes[1].hlines(sigr,0.01,1.19,color='green',linestyle='dashed')
axes[1].set_xlabel('B',fontsize=15)
axes[1].set_xlim([0,1.2])
axes[1].set_ylim([0.,0.6])
axes[1].tick_params(axis='x',labelsize=15)
axes[1].tick_params(axis='y',labelsize=15)
axes[1].legend(loc=9,fontsize=12)

axes[2].plot(dBT_dsig,sig2,label='dB/d$\sigma$')
axes[2].plot(BT,sig2,label='B')
axes[2].plot(dB2_dsig,sig2,label='$dB_2/sigma$')
axes[2].set_ylim([0.,0.6])
axes[2].tick_params(axis='x',labelsize=15)
axes[2].tick_params(axis='y',labelsize=15)
axes[2].legend(loc=9, ncol=2, fontsize=12)

plt.show()
