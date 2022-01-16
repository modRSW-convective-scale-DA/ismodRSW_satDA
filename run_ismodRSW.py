#######################################################################
### Script to integrate the ismodRSW model
#######################################################################
'''
Given mesh, IC, time paramters, integrates modRSW and plots evolution. 
Useful first check of simulations before use in the EnKF.
'''

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import matplotlib
matplotlib.use('Agg')
from builtins import str
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
import errno 
import os
import sys
import importlib.util
import random

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################
from f_ismodRSW import step_forward_ismodRSW, time_step, make_grid, make_grid_2
from isen_func import interp_sig2etab, interp_sig2etab_keq1, M_int, dMdsig, dsig_detab

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

Nk = config.Nk_tr
ic = config.ic
outdir = config.outdir
L = config.L
Neq = config.Neq
S0 = config.S0
A = config.A
V = config.V
R = config.R
theta1 = config.theta1
theta2 = config.theta2
eta0 = config.eta0
Z0 = config.Z0
g = config.g
U_scale = config.U_scale
table_file_name = config.table_file_name
sig_c = config.sig_c
sig_r = config.sig_r
tn = config.tn
Nmeas = config.Nmeas
tmax = config.tmax
dtmeasure = config.dtmeasure
tmeasure = config.tmeasure
cfl = config.cfl_tr
cc2 = config.cc2
alpha2 = config.alpha2
beta = config.beta
Ro = config.Ro
k = config.k
tau_rel = config.tau_rel
U_relax = config.U_relax

print(' -------------- ------------- ------------- ------------- ')
print(' --- TEST CASE: model only (dynamics and integration) --- ')
print(' -------------- ------------- ------------- ------------- ')
print(' ')
#print(' Number of elements Nk =', Nk)
print(' Initial condition:', ic)
print(' ')

#################################################################
### create directory for output
#################################################################
dirname = str('/nature_run')
dirn = str(outdir+dirname)
#check if dir exixts, if not make it
try:
    os.makedirs(dirn)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

##################################################################    
### Mesh generation for forecast and truth resolutions
##################################################################
grid =  make_grid(Nk,L) # forecast
Kk = grid[0]
x = grid[1]
xc = grid[2]

##################################################################    
### Apply initial conditions
##################################################################
print('Generate initial conditions...')
U0, B = ic(x,Nk,Neq,S0,L,A,V)

### 4 panel subplot for initial state of 4 vars
fig, axes = plt.subplots(4, 1, figsize=(8,8))
plt.suptitle("Initial condition with Nk = %d" %Nk)

axes[0].plot(xc, U0[0,:], 'b')
axes[0].set_ylabel('$\sigma_0(x)$',fontsize=18)
axes[0].plot(xc,sig_c*np.ones(len(xc)),'r:')
axes[0].plot(xc,sig_r*np.ones(len(xc)),'r:')

axes[1].plot(xc, U0[1,:]/S0, 'b')
axes[1].set_ylabel('$u_0(x)$',fontsize=18)

axes[2].plot(xc, U0[2,:]/S0, 'b')
axes[2].set_ylabel('$v_0(x)$',fontsize=18)

axes[3].plot(xc, U0[3,:]/S0, 'b')
axes[3].set_ylabel('$r_0(x)$',fontsize=18)
axes[3].set_xlabel('$x$',fontsize=18)

name_fig = "/ic_Nk%d.png" %Nk
f_name_fig = str(dirn+name_fig)
plt.savefig(f_name_fig)
print("** Initial condition "+str(name_fig)+" saved to "+ dirn)

### LOAD LOOK-UP TABLE
h5_file = h5py.File('inversion_tables/'+table_file_name,'r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

##################################################################
### Define system arrays and other indices
##################################################################
U = np.empty([Neq,Nk])
U_exact = np.zeros([Neq,Nk])
U_hovplt = np.zeros([Neq,Nk,120000])
t_hovplt = np.zeros(120000)
U = U0
U_array = np.empty((Neq,Nk,Nmeas+1))
U_ex = np.empty((Neq,Nk,Nmeas+1))
U_array[:,:,0] = U0
i = 0
p = 0
index = 1

##################################################################
### Convection and rain thresholds
##################################################################
etab_c = interp_sig2etab(sig_c,h5_file_data)
etab_r = interp_sig2etab(sig_r,h5_file_data)
Mc = M_int(etab_c,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)

##################################################################
### Relaxation solution
##################################################################
U_rel = U_relax(Neq,Nk,L,V,xc,U)

### PLOT AND STORE RELAXATION SOLUTION
fig, axes = plt.subplots(4, 1, figsize=(8,8))
plt.suptitle("Relaxation solution with Nk = %d" %Nk)

axes[0].plot(xc, U_rel[0,:], 'b',label='Relaxation solution')
axes[0].set_ylabel('$\sigma(x)$',fontsize=18)
axes[0].get_yaxis().set_label_coords(-0.1,0.5)
axes[0].set_xticklabels([])
axes[0].tick_params(labelsize=15)

axes[1].plot(xc, U_rel[1,:], 'r')
axes[1].set_ylabel('$u(x)$',fontsize=18)
axes[1].get_yaxis().set_label_coords(-0.1,0.5)
axes[1].set_xticklabels([])
axes[1].tick_params(labelsize=15)

a = axes[2].plot(xc, U_rel[2,:], 'b')
axes[2].set_ylabel('$v(x)$',fontsize=18)
axes[2].get_yaxis().set_label_coords(-0.1,0.5)
axes[2].set_xticklabels([])
axes[2].tick_params(labelsize=15)

axes[3].plot(xc, U_rel[3,:], 'r')
axes[3].set_ylabel('$r(x)$',fontsize=18)
axes[3].get_yaxis().set_label_coords(-0.1,0.5)
axes[3].set_xlabel('$x$',fontsize=18)
axes[3].tick_params(labelsize=15)

name_fig = "/U_rel_Nk%d.png" %Nk
f_name_fig = str(dirn+name_fig)
plt.savefig(f_name_fig)
print("** Relaxation solution "+str(name_fig)+" saved to "+ dirn)

print(' ')
print('Integrating forward from t =', round(tn,3), 'to', round(tmeasure,3),'...')

while tn < tmax:
 
    etab = interp_sig2etab(U[0,:],h5_file_data)
    dM_dsig = dMdsig(etab,U[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M = M_int(etab,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale) 

    dt = time_step(U,Kk,cfl,dM_dsig,cc2,beta)
    tn = tn + dt
#    print(tn) # comment out to print time

    U = step_forward_ismodRSW(U,U_rel,dt,tn,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,tau_rel) 

    # Save data for hovmoller plot
    if(p%3==0 and i<120000): 
        U_hovplt[:,:,i] = U  
        t_hovplt[i] = tn-dt
        i = i+1
    p = p+1
    
    if tn > tmeasure:
        
        print(' ')
        print('Plotting at time =',tmeasure)

        fig, axes = plt.subplots(4, 1, figsize=(8,8))
        plt.suptitle("Model trajectories at t = %.3f with Nk =%d" %(tmeasure,Nk))
        
        axes[0].plot(xc, U[0,:], 'b')
        axes[0].plot(xc,sig_c*np.ones(len(xc)),'r:')
        axes[0].plot(xc,sig_r*np.ones(len(xc)),'r:') 
        axes[0].set_ylabel('$\sigma(x)$',fontsize=18)
        axes[0].yaxis.set_label_coords(-0.1,0.5)        
 
        axes[1].plot(xc, U[1,:]/U[0,:], 'b')
        axes[1].set_ylabel('$u(x)$',fontsize=18)
        axes[1].yaxis.set_label_coords(-0.1,0.5)

        axes[2].plot(xc, U[2,:]/U[0,:], 'b')
        axes[2].set_ylabel('$v(x)$',fontsize=18)
        axes[2].yaxis.set_label_coords(-0.1,0.5)

        axes[3].plot(xc, U[3,:]/U[0,:], 'b')
        axes[3].plot(xc,np.zeros(len(xc)),'k--')
        axes[3].set_ylabel('$r(x)$',fontsize=18)
        axes[3].set_xlabel('$x$',fontsize=18)
        axes[3].yaxis.set_label_coords(-0.1,0.5)
        
        name_fig = "/t%d_Nk%d.png" %(index, Nk)
        f_name_fig = str(dirn+name_fig)
        plt.savefig(f_name_fig)
        plt.close()
        print(' *** %s at time level %d saved to %s' %(name_fig,index,dirn))
        
        U_array[:,:,index] = U
        U_ex[:,:,index] = U_exact        

        index = index + 1
        tmeasure = tmeasure + dtmeasure
        print(' ')
        print('Integrating forward from t =', round(tmeasure-dtmeasure,3), 'to', round(tmeasure,3),'...')
        
print(' ')
print('***** DONE: end of simulation at time:', tn)
print(' ')

print(' Saving simulation data in:', dirn)

np.save(str(outdir+'/U_array_Nk%d' %Nk),U_array)
np.save(str(outdir+'/U_hovplt'),U_hovplt)
np.save(str(outdir+'/t_hovplt'),t_hovplt)

print(' ')
print(' CHECK data value: maximum h(x) at t = 0.288:' , np.max(U_array[0,:,2]), ' at x = ', xc[np.argmax(U_array[0,:,2])])
print(' ')
print(' -------------- SUMMARY: ------------- ')  
print(' ') 
print('Dynamics:')
print('Ro =', Ro)  
print('(sig_c , sig_r) =', [sig_c, sig_r])  
print(' Mesh: number of elements Nk =', Nk)
print(' ')   
print(' ----------- END OF SUMMARY ---------- ')
print(' ')  


##################################################################
#			END OF PROGRAM				 #
##################################################################
