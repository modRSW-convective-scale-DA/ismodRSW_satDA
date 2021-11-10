import numpy as np
import sys
import importlib.util
import matplotlib.pyplot as plt
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
eta0 = config.eta0
U_scale = config.U_scale
g = config.g
Z0 = config.Z0
S0 = config.S0
Nk = config.Nk
L = config.L
kx1 = config.kx1
Ro = config.Ro

Kk = L/Nk

k = R/cp

eta_0 = interp_sig2etab(S0)
w1 = np.sqrt((1./Ro**2)+S0*kx1**2*k*(cp*theta2/U_scale**2)*eta_0**(k-1)*(1./dsig_detab(eta_0,0.,R,cp,theta1,theta2,eta0,g,Z0,U_scale)))

SIG = 0.0001

n = int(sys.argv[2])
tn = 0.072*n

xc = np.linspace(Kk/2,L-Kk/2,Nk)

U_tr = np.load(outdir+str('/U_array_Nk'+str(Nk)+'.npy'))
sig_num = U_tr[0,:,n]
sig_exact = S0+SIG*np.cos(kx1*xc-w1*tn)

err_inf = np.amax(sig_num-sig_exact)

print(err_inf)
