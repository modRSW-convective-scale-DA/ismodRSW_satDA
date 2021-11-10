import numpy as np
import sys
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
from isen_func import *

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
sig_c = config.sig_c
cp = config.cp
R = config.R
theta1 = config.theta1
theta2 = config.theta2
U = config.U_scale
eta0 = config.eta0
g = config.g
Z0 = config.Z0
S0 = config.S0
pref = config.pref
Nk = config.Nk
kx1 = config.kx1
Neq = config.Neq
Ro = config.Ro

Kk = 1./Nk

t = int(sys.argv[2])

def kin_energy(sigma,u,v):

    KE_x = 0.5*sigma*u**2
    KE_y = 0.5*sigma*v**2
    KE = KE_x+KE_y

    return KE, KE_x, KE_y

def pot_energy(sigma):

    k = R/cp
    kinv = 1./k
    etab = interp_sig2etab(sigma)
    eta1 = ((theta2/(theta1-theta2))**kinv)*(-etab**k+(theta1/theta2)*eta0**k+g*Z0/(cp*theta2))**kinv
    etac = interp_sig2etab(sig_c)
    PE_const = ((theta2/(theta1-theta2))**kinv)*(k/(k+1))*((theta1/theta2)*eta0**k+g*Z0/(cp*theta2))**(kinv+1)
    PE_1 = (etab**(k+1.))/(k+1.)
    PE_2 = eta1*etab**k
    PE_3 = (k/(k+1))*((theta1-theta2)/theta2)*eta1*eta1**k
    PE = (cp*theta2/U**2)*(PE_1-PE_2-PE_3)
    
    return PE,PE_const,PE_1,PE_2,PE_3

U_tr = np.load(outdir+str('/U_array_Nk'+str(Nk)+'.npy'))
U_exact = np.zeros((Neq-1,Nk))

KE = np.empty(np.size(U_tr,1))
KE_ex = np.empty(np.size(U_tr,1))
KE_x = np.empty(np.size(U_tr,1))
KE_exx = np.empty(np.size(U_tr,1))
KE_y = np.empty(np.size(U_tr,1))
KE_exy = np.empty(np.size(U_tr,1))

PE = np.empty(np.size(U_tr,1))
PE_ex = np.empty(np.size(U_tr,1))
PE_1 = np.empty(np.size(U_tr,1))
PE_ex1 = np.empty(np.size(U_tr,1))
PE_2 = np.empty(np.size(U_tr,1))
PE_ex2 = np.empty(np.size(U_tr,1))
PE_3 = np.empty(np.size(U_tr,1))
PE_ex3 = np.empty(np.size(U_tr,1))
PE_const = np.empty(np.size(U_tr,1))

k = R/cp
SIG = 0.0001
xc = np.linspace(Kk/2.,1.-Kk/2.,Nk)
eta_0 = interp_sig2etab(S0)
w1 = np.sqrt((1./Ro**2)+S0*kx1**2*k*(cp*theta2/U**2)*eta_0**(k-1)*(1./dsig_detab(eta_0,0.,R,cp,theta1,theta2,eta0,g,Z0,U)))
tn = t*0.072

U_exact[0,:] = S0+SIG*np.cos(kx1*xc-w1*tn)
U_exact[1,:] = SIG*(w1/(S0*kx1))*np.cos(kx1*xc-w1*tn)
U_exact[2,:] = SIG*(1./(Ro*S0*kx1))*np.sin(kx1*xc-w1*tn)
KE_ex,KE_exx,KE_exy = kin_energy(U_exact[0,:],U_exact[1,:],U_exact[2,:])
KE,KE_x,KE_y = kin_energy(U_tr[0,:,t],U_tr[1,:,t]/U_tr[0,:,t],U_tr[2,:,t]/U_tr[0,:,t])
PE_ex,PE_const,PE_ex1,PE_ex2,PE_ex3 = pot_energy(U_exact[0,:])
PE,PE_const,PE_1,PE_2,PE_3 = pot_energy(U_tr[0,:,t])
 
fig, axes = plt.subplots(3, 1, figsize=(8,8))
axes[0].set_ylabel('Energy')
axes[0].set_xlabel('Domain')
axes[0].plot(list(range(np.size(U_tr,1))),np.cumsum(KE),color='red')
axes[0].plot(list(range(np.size(U_tr,1))),np.cumsum(KE_x),color='blue')
axes[0].plot(list(range(np.size(U_tr,1))),np.cumsum(KE_y),color='orange')
axes[0].plot(list(range(np.size(U_tr,1))),np.cumsum(KE_ex),color='green')
#axes[0].set_ylim([1.06e-5,1.12e-5])
KE_tot_line = mlines.Line2D([],[],color='red',label='Total KE')
KE_hor_line = mlines.Line2D([],[],color='blue',label='Horizontal KE')
KE_mer_line = mlines.Line2D([],[],color='orange',label='Meridional KE')
axes[0].legend(handles=[KE_tot_line,KE_hor_line,KE_mer_line],loc=9)
axes[1].set_ylabel('Energy')
axes[1].set_xlabel('Time')
#axes[1].plot(list(range(np.size(U_tr,2))),PE_1+PE_2,color='black')A
axes[1].plot(list(range(np.size(U_tr,1))),abs(np.cumsum(PE_ex))-abs(np.cumsum(PE)),color='red')
#axes[1].plot(list(range(np.size(U_tr,1))),abs(np.cumsum(PE_ex)),color='green')
#axes[1].set_yscale("log")
#axes[1].set_ylim([2e4,2.1e4])
#axes[1].plot(list(range(np.size(U_tr,2))),,color='red')
#PE_tot_line = mlines.Line2D([],[],color='red',label='Total PE')
#axes[1].legend(handles=[PE_tot_line],loc=9)
#axes[1].set_ylim(1.9770e9,1.97720e9)
#axes[1].plot(range(np.size(U_tr,2)),PE_2,color='blue')
#axes[1].plot(range(np.size(U_tr,2)),PE,color='red')
#ax2 = axes[1].twinx()
#axes[1].plot(range(np.size(U_tr,2)),PE_1,color='green')
#axes[1].plot(range(np.size(U_tr,2)),PE_3,color='black')
#axes[1].plot(range(np.size(U_tr,2)),PE_const,color='orange')
#axes[2].plot(list(range(np.size(U_tr,2))),PE+KE,color='red')
axes[2].plot(list(range(np.size(U_tr,1))),np.cumsum(PE_ex)+np.cumsum(KE_ex),color='green')
plt.show()
#plt.savefig('/nobackup/mmlca/figs/truth_trans_jet+hor_vel.png')
