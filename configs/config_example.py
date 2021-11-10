####################################################################
##  FILE CONTAINING CONFIGURATION PARAMENTERS            	  ##
####################################################################
'''
List of fixed parameters for model integration and EnKF.
'''

import numpy as np
from init_cond_isenRSW import init_cond_9
from relax_sol_isenRSW import U_relax_12

'''Output directory'''
outdir = '/nobackup/mmlca/isen_enkf/2xres_1h/config#325'

''' MODEL PARAMETERS '''

Neq = 4     # number of equations in system (3 with topography, 4 with rotation_
L = 1.0     # length of domain (non-dim.)

Nk_fc = 200                                 # forecast resolution
dres = 2                                     # refinement factor for truth gridsize
Nk_tr = dres*Nk_fc                           # truth resolution
n_d = Neq * Nk_fc                   # total number of variables (dgs of freedom)
Kk_fc = L/Nk_fc
Kk_tr = L/Nk_tr

cfl_fc = 0.1 # Courant Friedrichs Lewy number for time stepping
cfl_tr = 0.1

Ro = 0.248          # Rossby no. Ro ~ V0/(f*L0)
g = 9.81 	    # gravity acceleration
A = 0.04
V = 1.

theta1 = 311.                                                   # top-layer potential temperature
theta2 = 291.8                                                   # bottom-layer potential temperature
Z0 = 6120.                                                     # total depth
eta0 = 0.48                                                     # pressure at the top of the layers
pref = 1000.                                                    # pressure reference at the surface
R = 287.                                                        # ideal gas constant for dry air
cp = 1004.                                                      # constant-pressure heat coefficient
k = R/cp
U_scale = 12.4

# RELAXATION SOLUTION
U_relax = U_relax_12
tau_rel = 4.

# CHOOSE INITIAL CONDITION FROM init_cond_modRSW: 
ic = init_cond_9

# threshold heights
S0 = 0.2
sig_c = 0.21
sig_r = 0.24

# remaining parameters related to hr
beta = 2.
alpha2 = 6.
cc2 = 1.8

''' FILTER PARAMETERS '''

n_ens = 20                              # number of ensembles
TIMEOUT_PAR = n_ens*8			# time to wait until all forecasts running in parallel are over
Nmeas = 96                              # number of cycles
Nforec = 7				# duration of each forecast in dtmeasure
NIAU = 1000				# suppress injection of additional inflation with IAU for the first NIAU hours since assimilation 
IAU_dir = ''
tn = 0.0                                # initial time
spin_up = 12
dtmeasure = 0.089  
tmax = (Nmeas+Nforec)*dtmeasure
t_end_assim = Nmeas*dtmeasure
tmeasure = dtmeasure
assim_time = np.linspace(tn,t_end_assim,Nmeas+1) # vector of times when system is observed
lead_times = [0,3,6]
ass_freq='1h'

### INITIAL CONDITION
sig_ic = [0.02,0.008,0.1,0.0]

### Q MATRIX GENERATION
model_noise = [0.0,0.0,0.0,0.0]
Nhr = 1 # parameter for Q_nhr
Q_FUNC = 'Q_nhr()' ### Q_predef(), Q_nhr()
rMODNOISE = 1
sigMODNOISE = 0

### OBSERVING SYSTEM
# Observation properties
ob_noise = [0.01,0.0001,0.04,0.04,0.01]            # ob noise for [sat,T,sigu,sigv,sigr]
o_d = 20                                # ob density: observe every o_d elements
n_d = Neq*Nk_fc
swathe = np.array([20,20,40,40,60,60,80,80])                # width of the satellite's swath (in Km)
sat_init_pos = np.array([0.1,0.64,0.17,0.75,0.35,0.88,0.41,0.9]) # position of satellites at first analysis step
sat_vel = np.array([0.2,-0.12,-0.25,0.18,0.15,-0.15,0.1,-0.1])
n_obs_sat = len(swathe)
obs_T_d = 1000
obs_u_d = 20
obs_v_d = 20
obs_r_d = 20
n_obs_T = Nk_fc // obs_T_d
n_obs_u = Nk_fc // obs_u_d
n_obs_v = Nk_fc // obs_v_d
n_obs_r = Nk_fc // obs_r_d
n_obs_grnd = n_obs_T+n_obs_u+n_obs_v+n_obs_r
n_d_grnd = Neq*Nk_fc                  # number of ground variables (dgs of freedom)
n_obs = n_obs_grnd + n_obs_sat
sat_obs_mask = np.array(list(range(n_obs_sat)))
sigT_obs_mask = np.array(list(range(n_obs_sat,n_obs_sat+n_obs_T)))
sigu_obs_mask = np.array(list(range(n_obs_sat+n_obs_T,n_obs_sat+n_obs_T+n_obs_u)))
sigv_obs_mask = np.array(list(range(n_obs_sat+n_obs_T+n_obs_u,n_obs_sat+n_obs_T+n_obs_u+n_obs_v)))
sigr_obs_mask = np.array(list(range(n_obs_sat+n_obs_T+n_obs_u+n_obs_v,n_obs_sat+n_obs_T+n_obs_u+n_obs_v+n_obs_r)))
sim_title = 'control'
config_str = 'con'

''' OUTER LOOP'''
'''
Parameters for outer loop are specified in main_p.py 
loc     : localisation scale
add_inf : additive infaltaion factor
rtpp    : Relaxation to Prior Perturbations scaling factor
rtps    : Relaxation to Prior Spread scaling factor
'''
# MUST BE FLOATING POINT
loc = [1.0]
add_inf = [0.5]
rtpp = [0.5]
rtps = [0.6]

