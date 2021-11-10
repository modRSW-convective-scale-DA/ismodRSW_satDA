import numpy as np
import matplotlib.pyplot as plt

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
sigc = pref*0.211/g
sigr = pref*0.24/g

eta2 = np.linspace(0.9,1.1,100)

#### Sigma2 equation
sig2 = (pref/g)*(eta2-((1./cp)*(1./(theta1-theta2))*(-cp*theta2*eta2**k+cp*theta1*eta0**k+g*Z0))**kinv)

#### Eta1 equation
eta1 = eta2-g*sig2/pref

#### Layer thickness
h1 = (cp*theta1/g)*(eta1**k-eta0**k)
h2 = (cp*theta2/g)*(eta2**k-eta1**k)

#### Temperature definition
T1 = theta1*eta1**k
T2 = theta2*eta2**k

#### Plots
fig, axes = plt.subplots(1, 3, figsize=(13,5))
axes[0].plot(eta2,sig2,label='$\eta_2$')
axes[0].plot(eta1,sig2,label='$\eta_1$')
axes[0].hlines(sigc,0.1,1.1,color='red',linestyle='dashed',label='$\sigma_c$')
axes[0].hlines(sigr,0.1,1.1,color='green',linestyle='dashed',label='$\sigma_r$')
axes[0].set_xlabel('$\eta$',fontsize=15)
axes[0].set_ylabel('$\sigma_2$',fontsize=15)
axes[0].set_ylim([0.,90.])
axes[0].set_xlim([0.00,1.2])
axes[0].tick_params(axis='x',labelsize=15)
axes[0].tick_params(axis='y',labelsize=15)
axes[0].legend(loc=9,ncol=2,fontsize=12)

axes[1].plot(T2,sig2,label='$T_2$')
axes[1].plot(T1,sig2,label='$T_1$')
axes[1].hlines(sigc,220,320,color='red',linestyle='dashed',label='$\sigma_c$')
axes[1].hlines(sigr,220,320,color='green',linestyle='dashed',label='$\sigma_r$')
axes[1].set_xlabel('$T$ (K)',fontsize=15)
axes[1].set_xlim([210.,330.])
axes[1].set_ylim([10.,60.])
axes[1].tick_params(axis='x',labelsize=15)
axes[1].tick_params(axis='y',labelsize=15)
axes[1].legend(loc=9,ncol=2,fontsize=12)

axes[2].plot(h2,sig2,label='$h_2$')
axes[2].plot(h1,sig2,label='$h_1$')
axes[2].hlines(sigc,100,6200,color='red',linestyle='dashed',label='$\sigma_c$')
axes[2].hlines(sigr,100,6200,color='green',linestyle='dashed',label='$\sigma_r$')
axes[2].set_xlabel('$h$ (m)',fontsize=15)
axes[2].set_ylim([10.,60.])
axes[2].set_xlim([0.,6200])
axes[2].tick_params(axis='x',labelsize=15)
axes[2].tick_params(axis='y',labelsize=15)
axes[2].legend(loc=9,ncol=2,fontsize=12)

plt.show()
