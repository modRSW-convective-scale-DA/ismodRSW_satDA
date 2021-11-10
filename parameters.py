from builtins import range
import numpy as np

####################################################################
##                       FIXED PARAMETERS                            ##
####################################################################

'''Output directory'''
outdir = '/nobackup/mmlca/test_isen_enkf'

Neq = 4                                                         # number of equations
L = 1. 								# lenght of domain (non-dim)

Nk_fc = 200							# forecast resolution
Kk_fc = L/Nk_fc
dres = 4							# refinment factor for truth gridsize
Nk_tr = dres*Nk_fc						# truth resolution

cfl = 0.5							# Courant Friedrichs Lewy number for time stepping
Ro = 0.1                                                        # Rossby number
g = 9.81                                                        # gravity acceleration
S0 = 0.2							# pseudo-density initial state
A = 0.05							
V = 1.								
theta1 = 310.                                                   # top-layer potential temperature
theta2 = 290.                                                   # bottom-layer potential temperature
Z0 = 11000.                                                     # total depth
eta0 = 0.22                                                     # pressure at the top of the layers
pref = 1000.                                                    # pressure reference at the surface
R = 287.                                                        # ideal gas constant for dry air
cp = 1004.                                                      # constant-pressure heat coefficient
U = 20.

# threshold (for pseudo-density)
sig_c = 0.94
sig_r = 0.95

# Rain and convection parameters
alpha2 = 5.
beta = 0.4
cc2 = 0.05

''' FILTER PARAMETERS '''

sig_ic = [0.01,0.04,0.04,0.0]                # initial ens perturbations [h,hu,hr]
n_ens = 10                              			# number of ensembles
Nmeas_DA = 3	                           			# number of DA cycles
Nmeas_su = 12    	                       			# number of spin-up cycles
tn = 0.0                                			# initial time
dtmeasure = 0.072
tspinup = dtmeasure*Nmeas_su
t_DA = dtmeasure*Nmeas_DA
tmax = tspinup + t_DA
tmeasure = dtmeasure
Nmeas = Nmeas_DA+Nmeas_su
assim_time = np.linspace(tn,tmax,Nmeas+1) 			# vector of times when system is observed

# Observation properties
ob_noise = [0.01,0.05,0.05,0.005]            # ob noise for [sig,sigu,sigv,sigr]
o_d = 40                                # ob density: observe every o_d elements
n_d = Neq*Nk_fc
n_d_grnd = (Neq-1)*Nk_fc                  # number of ground variables (dgs of freedom)
n_obs_grnd = n_d_grnd//o_d                 # no. of ground observations
n_obs_sat = 1
swathe = 40				# width of the satellite's swath (in Km)
n_obs = n_obs_grnd + n_obs_sat
sig_obs_mask = np.array(list(range(n_obs_sat)))
sigu_obs_mask = np.array(list(range(n_obs_sat,n_obs_sat+n_obs_grnd//(Neq-1))))
sigv_obs_mask = np.array(list(range(n_obs_sat+n_obs_grnd//(Neq-1),n_obs_sat+2*n_obs_grnd//(Neq-1))))
sigr_obs_mask = np.array(list(range(n_obs_sat+2*n_obs_grnd//(Neq-1),n_obs_sat+3*n_obs_grnd//(Neq-1))))

#

''' OUTER LOOP'''
'''
Parameters for outer loop are specified in main_p.py 
loc     : localisation scale
add_inf : additive infaltaion factor
rtpp    : Relaxation to Prior Perturbations scaling factor
rtps    : Relaxation to Prior Spread scaling factor
'''

# MUST BE FLOATING POINTS
loc = [ 3.]
add_inf = [ 0.05]
rtpp = [0.5]
rtps = [ 0.7]
