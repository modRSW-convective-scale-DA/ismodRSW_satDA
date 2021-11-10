#######################################################################
# isenRSW with topography
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
from f_isenRSW import step_forward_isenRSW, time_step, make_grid, make_grid_2
from isen_func import interp_sig2etab, interp_sig2etab_keq1, M_int, dMdsig, dsig_detab, rossdef_radius

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

Nk = config.Nk
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
sig_c = config.sig_c
sig_r = config.sig_r
tn = config.tn
Nmeas = config.Nmeas
tmax = config.tmax
dtmeasure = config.dtmeasure
tmeasure = config.tmeasure
cfl = config.cfl
cc2 = config.cc2
alpha2 = config.alpha2
beta = config.beta
Ro = config.Ro
#kx1 = config.kx1
#SIG = config.SIG
k = config.k
tau_rel = config.tau_rel
U_relax = config.U_relax
acc = config.acc

print(' -------------- ------------- ------------- ------------- ')
print(' --- TEST CASE: model only (dynamics and integration) --- ')
print(' -------------- ------------- ------------- ------------- ')
print(' ')
#print(' Number of elements Nk =', Nk)
print(' Initial condition:', ic)
print(' ')

#################################################################
# create directory for output
#################################################################
dirname = str('/test_model')
dirn = str(outdir+dirname)
#check if dir exixts, if not make it
try:
    os.makedirs(dirn)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

##################################################################    
# Mesh generation for forecast and truth resolutions
##################################################################
grid =  make_grid(Nk,L) # forecast
Kk = grid[0]
x = grid[1]
xc = grid[2]

##################################################################    
#%%%----- Apply initial conditions -----%%%
##################################################################
#kinv = 1./k
#eta_0 = interp_sig2etab(S0)
#w1 = np.sqrt((1./Ro**2)+S0*kx1**2*k*((R*kinv)*theta2/U_scale**2)*eta_0**(k-1)*(1./dsig_detab(eta_0,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)))

#LD = rossdef_radius(Ro,S0,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)

print('Generate initial conditions...')
U0, B = ic(x,Nk,Neq,S0,L,A,V)

#U0[0,:] = S0+SIG*np.cos(kx1*x[:-1])
#U0[1,:] = SIG*(w1/(S0*kx1))*np.cos(kx1*x[:-1])*U0[0,:]
#U0[2,:] = SIG*(1./(Ro*S0*kx1))*np.sin(kx1*x[:-1])*U0[0,:]

'''
shrira_data = np.load('/home/home02/mmlca/isenRSW/data_shrira_sol_3per.npy')

xc = shrira_data[0]
Kk = xc[1] - xc[0]
Nk = len(xc)

U0 = np.zeros((Neq,Nk))
U0[0,:] = shrira_data[1]
U0[1,:] = shrira_data[2]*U0[0,:]
U0[2,:] = shrira_data[3]*U0[0,:]
'''
### 4 panel subplot for initial state of 4 vars
fig, axes = plt.subplots(4, 1, figsize=(8,8))
plt.suptitle("Initial condition with Nk = %d" %Nk)

axes[0].plot(xc, U0[0,:], 'b')
#axes[0].plot(xc, B, 'k', linewidth=2.)
axes[0].set_ylabel('$\sigma_0(x)$',fontsize=18)
#axes[0].plot(xc,sig_c*np.ones(len(xc)),'r:')
#axes[0].plot(xc,sig_r*np.ones(len(xc)),'r:')
#axes[0].set_ylim([0,2])

axes[1].plot(xc, U0[1,:]/S0, 'b')
#axes[1].set_ylim([-2,2])
axes[1].set_ylabel('$u_0(x)$',fontsize=18)

axes[3].plot(xc, U0[2,:]/S0, 'b')
#axes[].set_ylim([-2,2])
axes[3].set_ylabel('$v_0(x)$',fontsize=18)

axes[2].plot(xc, U0[3,:]/S0, 'b')
axes[2].set_ylabel('$r_0(x)$',fontsize=18)
#axes[2].set_ylim([-0.025,0.15])
axes[2].set_xlabel('$x$',fontsize=18)

#plt.show() # use block=False?
#exit()
name_fig = "/ic_Nk%d.png" %Nk
f_name_fig = str(dirn+name_fig)
plt.savefig(f_name_fig)
print("** Initial condition "+str(name_fig)+" saved to "+ dirn)
#plt.close()

### LOAD LOOK-UP TABLE
h5_file = h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_291_theta1_311.hdf','r')
h5_file_data = h5_file.get('sigma_eta_iversion_table')[()]

##################################################################
#'''%%%----- Define system arrays and time parameters------%%%'''
##################################################################
U = np.empty([Neq,Nk])
U_exact = np.zeros([Neq,Nk])
U_hovplt = np.zeros([Neq,Nk,120000])
t_hovplt = np.zeros(120000)
U = U0

index = 1

### Convection and rain thresholds
#etab_c = interp_sig2etab(sig_c,h5_file_data)
#etab_c = 1000.
#etab_r = interp_sig2etab(sig_r,h5_file_data)
#etab_r = 1000.
etab_c = interp_sig2etab_keq1(sig_c,theta1,theta2,eta0,g,R,Z0)
etab_r = interp_sig2etab_keq1(sig_r,theta1,theta2,eta0,g,R,Z0)
Mc = M_int(etab_c,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)

##################################################################
#'''%%%----- integrate forward in time until tmax ------%%%'''
##################################################################
U_array = np.empty((Neq,Nk,Nmeas+1))
U_ex = np.empty((Neq,Nk,Nmeas+1))
U_array[:,:,0] = U0
i = 0
p = 0

### Relaxation solution ###
U_rel = U_relax(Neq,Nk,L,V,xc,U)
#U_rel = np.zeros((Neq,Nk))

#Lj = 0.1*L
#f1 = V*(1+np.tanh(4*(xc-0.2*L)/Lj + 2))*(1-np.tanh(4*(xc-0.2*L)/Lj - 2))/4
#f2 = V*(1+np.tanh(4*(xc-0.4*L)/Lj + 2))*(1-np.tanh(4*(xc-0.4*L)/Lj - 2))/4
#f3 = V*(1+np.tanh(4*(xc-0.6*L)/Lj + 2))*(1-np.tanh(4*(xc-0.6*L)/Lj - 2))/4
#f4 = V*(1+np.tanh(4*(xc-0.8*L)/Lj + 2))*(1-np.tanh(4*(xc-0.8*L)/Lj - 2))/4
#ic3 = f1-f2+f3-f4
#U_rel[2,:] = ic3*U[0,:]

### PLOT AND STORE RELAXATION SOLUTION
fig, axes = plt.subplots(4, 1, figsize=(8,8))
#plt.suptitle("Relaxation solution with Nk = %d" %Nk)

#axes[0].plot(xc, U_rel[0,:], 'b',label='Relaxation solution')
axes[0].plot(xc, U0[0,:], 'r')
axes[0].set_ylabel('$\sigma(x)$',fontsize=18)
axes[0].get_yaxis().set_label_coords(-0.1,0.5)
axes[0].set_xticklabels([])
axes[0].tick_params(labelsize=15)

axes[1].plot(xc, U0[1,:], 'r')
axes[1].set_ylabel('$u(x)$',fontsize=18)
axes[1].get_yaxis().set_label_coords(-0.1,0.5)
axes[1].set_xticklabels([])
axes[1].tick_params(labelsize=15)

a = axes[2].plot(xc, U_rel[2,:], 'b')
b = axes[2].plot(xc, U0[2,:]/S0, 'r')
axes[2].set_ylabel('$v(x)$',fontsize=18)
axes[2].get_yaxis().set_label_coords(-0.1,0.5)
axes[2].set_xticklabels([])
axes[2].tick_params(labelsize=15)

axes[3].plot(xc, U0[3,:], 'r')
axes[3].set_ylabel('$r(x)$',fontsize=18)
axes[3].get_yaxis().set_label_coords(-0.1,0.5)
axes[3].set_xlabel('$x$',fontsize=18)
axes[3].tick_params(labelsize=15)

fig.legend((a[0],b[0]),('Relaxation solution','Initial condition'),loc=9,fontsize=18)

#plt.show()
#exit()
name_fig = "/U_rel_Nk%d.png" %Nk
f_name_fig = str(dirn+name_fig)
plt.savefig(f_name_fig)
print("** Relaxation solution "+str(name_fig)+" saved to "+ dirn)
#plt.close()

print(' ')
print('Integrating forward from t =', round(tn,3), 'to', round(tmeasure,3),'...')

#dt = 5.28e-05

while tn < tmax:
 
    #etab = interp_sig2etab(U[0,:],h5_file_data)
    etab = interp_sig2etab_keq1(U[0,:],theta1,theta2,eta0,g,R,Z0)
    dM_dsig = dMdsig(etab,U[0,:],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
    M = M_int(etab,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale) 

   # dt_old = np.copy(dt)
    
    dt = time_step(U,Kk,cfl,dM_dsig,cc2,beta)
    tn = tn + dt
    print(tn) 

 #   dtmin = min(dt_old,dt)

#    U_infl = [S0+SIG*np.cos(kx1*0.-w1*tn),(w1/(S0*kx1))*SIG*np.cos(kx1*0.-w1*tn),(1./(Ro*S0*kx1))*SIG*np.sin(kx1*0.-w1*tn)]
#    U_oufl = [S0+SIG*np.cos(kx1*1.-w1*tn),(w1/(S0*kx1))*SIG*np.cos(kx1*1.-w1*tn),(1./(Ro*S0*kx1))*SIG*np.sin(kx1*1.-w1*tn)]
#    U_infl[1] = U_infl[0]*U_infl[1]
#    U_infl[2] = U_infl[0]*U_infl[2]
#    U_oufl[1] = U_oufl[1]*U_oufl[0]
#    U_oufl[2] = U_oufl[2]*U_oufl[0]
#    etab_infl = interp_sig2etab(U_infl[0])   
#    etab_oufl = interp_sig2etab(U_oufl[0])
#    M_infl = M_int(etab_infl,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
#    M_oufl = M_int(etab_oufl,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
#    dM_dsig_infl = dMdsig(etab_infl,U_infl[0],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
#    dM_dsig_oufl = dMdsig(etab_oufl,U_infl[0],0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)

    #print(U_infl[2]/U_infl[0])
   
    #if(p==0):
    #    dur = int(dtmeasure//dt) # minumum duration of the perturbation in time steps
    #    p = random.randint(dur,2*dur) # generate a random duration in time steps
    #    xn = round(random.random(),2) # generate a random position
    #    Fn = conv_pert_dens(Nk,xc,x,sig_pert,l_pert) # generate the perturbation
    #    Fn = conv_pert(Nk,Kk,xc,xn,U_pert,l_pert) # generate the perturbation
    #    Fn = Fn/p # split the perturbation in k smaller perturbations
   
    #U[1,:] = (U[1,:]+Fn)*U[0,:]
    #U[0,:] = U[0,:]+Fn
 
#    if tn > tmeasure:
#       	dt = dt-(tn-tmeasure)+1e-12
#        tn = tmeasure+1e-12    

    U = step_forward_isenRSW(U,U_rel,dt,tn,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,tau_rel) 
#    U = step_forward_isenRSW_inflow(U,dt,tn,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,BC,S0,w1,U_infl,M_infl,dM_dsig_infl)
#    U = step_forward_isenRSW_inflow_outflow(U,dt,tn,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,Ro,BC,S0,w1,U_infl,U_oufl,M_infl,M_oufl,dM_dsig_infl,dM_dsig_oufl)
    #U = step_forward_RK3(U,dt,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro,S0,w1,U_infl,M_infl,dM_dsig_infl)
#    U = step_forward_RK3_oufl(U,dt,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro,S0,w1,U_infl,M_infl,dM_dsig_infl,U_oufl,M_oufl,dM_dsig_oufl)
#    U = step_forward_periodic_RK3(U,dt,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro)
    #U = step_forward_periodic_RK3(U,dt,Nk,Neq,Kk,M,Mc,sig_r,sig_c,dM_dsig,cc2,alpha2,beta,R,k,theta1,theta2,eta0,g,Z0,U_scale,Ro)


 #   U_exact[0,:] = S0+SIG*np.cos(kx1*xc-w1*tn)
 #   U_exact[1,:] = SIG*(w1/(S0*kx1))*np.cos(kx1*xc-w1*tn)
 #   U_exact[2,:] = SIG*(1./(Ro*S0*kx1))*np.sin(kx1*xc-w1*tn) 

    #print(U_exact[2,0])

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
  #      axes[0].plot(xc, U_exact[0,:], 'r')
#        axes[0].plot(xc, B, 'k', linewidth=2.0)
    #    axes[0].plot(xc,sig_c*np.ones(len(xc)),'r:')
    #    axes[0].plot(xc,sig_r*np.ones(len(xc)),'r:') 
#        axes[0].set_ylim([0.,max(U[0,:]))
        axes[0].set_ylabel('$\sigma(x)$',fontsize=18)
        axes[0].yaxis.set_label_coords(-0.1,0.5)        
 
        axes[1].plot(xc, U[1,:]/U[0,:], 'b')
   #     axes[1].plot(xc, U_exact[1,:], 'r')
   #     axes[1].set_ylim([0,3])
        axes[1].set_ylabel('$u(x)$',fontsize=18)
        axes[1].yaxis.set_label_coords(-0.1,0.5)

        axes[3].plot(xc, U[2,:]/U[0,:], 'b')
    #    axes[3].plot(xc, U_exact[2,:], 'r')  
        #axes[2].plot(xc, B, 'k', linewidth=2.0)
        #axes[2].set_ylim([0,1300])
        axes[3].set_ylabel('$v(x)$',fontsize=18)
        axes[3].yaxis.set_label_coords(-0.1,0.5)

        axes[2].plot(xc, U[3,:]/U[0,:], 'b')
     #   axes[2].plot(xc, U_exact[3,:], 'r')
        axes[2].plot(xc,np.zeros(len(xc)),'k--')
        axes[2].set_ylabel('$r(x)$',fontsize=18)
  #      axes[2].set_ylim([-0.025,0.3])
        axes[2].set_xlabel('$x$',fontsize=18)
        axes[2].yaxis.set_label_coords(-0.1,0.5)
        
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
#np.save(str(outdir+'/U_exact_Nk%d' %Nk),U_ex)
#np.save(str(outdir+'/B_Nk%d' %Nk),B)

#fig, axes = plt.subplots(4, 1, figsize=(8,8))
#axes[0].plot(range(0,i),uplusc0)
#axes[0].set_title('U+c at x=0')
#axes[1].plot(range(0,i),upluscL)
#axes[1].set_title('U+c at x=L')
#axes[2].plot(range(0,i),uminsc0)
#axes[2].set_title('U-c at x=0')
#axes[3].plot(range(0,i),uminscL)
#axes[3].set_title('U-c at x=L')
#name_fig = "/uplusminsc.png"
#f_name_fig = str(dirn+name_fig)
#plt.savefig(f_name_fig)

print(' ')
print(' CHECK data value: maximum h(x) at t = 0.288:' , np.max(U_array[0,:,2]), ' at x = ', xc[np.argmax(U_array[0,:,2])])
print(' ')
print(' -------------- SUMMARY: ------------- ')  
print(' ') 
print('Dynamics:')
print('Ro =', Ro)  
#print 'Fr = ', Fr
print('(sig_c , sig_r) =', [sig_c, sig_r])  
print(' Mesh: number of elements Nk =', Nk)
#print(' Minimum time step = ', dtmin)
print(' ')   
print(' ----------- END OF SUMMARY ---------- ')
print(' ')  


##################################################################
#			END OF PROGRAM				 #
##################################################################

