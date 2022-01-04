####################################################################
##  FILE CONTAINING CONFIGURATION PARAMENTERS            	  ##
####################################################################
'''
List of fixed parameters for model integration and EnKF.
'''

import numpy as np
from init_cond_ismodRSW import init_cond_9
from relax_sol_ismodRSW import U_relax_12

'''Output directory'''
outdir = 'output/config#325'

''' MODEL PARAMETERS '''

Neq = 4                             # number of equations in system (3 with topography, 4 with rotation)
L = 1.0                             # length of domain (non-dimensional)

Nk_fc = 200                         # forecast resolution
dres = 2                            # refinement factor for truth gridsize
Nk_tr = dres*Nk_fc                  # truth resolution
n_d = Neq * Nk_fc                   # total number of variables (dgs of freedom)
Kk_fc = L/Nk_fc                     # non-dimensional grid step (forecast)
Kk_tr = L/Nk_tr                     # non-dimensional grid step (truth)

cfl_fc = 0.1                        # Courant Friedrichs Lewy number for time stepping (forecast)
cfl_tr = 0.1                        # Courant Friedrichs Lewy numper for time stepping (truth)

Ro = 0.248                          # Rossby number, i.e. Ro ~ V0/(f*L0)
g = 9.81 	                    # gravity acceleration [m/s]
A = 0.04                            # pseudo-density wave amplitude (used for initial condition)
V = 1.                              # velocity wave amplitude (used for initial condition)

theta1 = 311.                       # top-layer potential temperature [K]
theta2 = 291.8                      # bottom-layer potential temperature [K]
Z0 = 6120.                          # total fluid depth [m]
eta0 = 0.48                         # non-dimensional pressure at the top of the layers [p/pref]
pref = 1000.                        # pressure reference at the surface [hPa]
R = 287.                            # specific gas constant for dry air [J/(kg*K)]
cp = 1004.                          # isobaric mass heat capacity for dry air [J/(Kg*K)]
k = R/cp                            
U_scale = 12.4                      # scaling velocity for the bottom layer [m/s]

# LOOK-UP TABLE file name
table_file_name = 'sigma_eta_theta2_'+str(int(theta2))+'_theta1_'+str(int(theta1))+'_eta0_'+str(eta0)+'_Z0_'+str(int(Z0))+'_k_'+str(round(k,2))+'.hdf'

# RELAXATION SOLUTION FROM relax_sol_ismodRSW.py:
U_relax = U_relax_12
tau_rel = 4.

# PSEUDO-DENSITY THRESHOLDS
S0 = 0.2                            # constant 'at rest' pseudo-density
sig_c = 0.21                        # convection pseudo-density constant
sig_r = 0.24                        # rain pseudo-density constant

# RAIN AND CONVECTION PARAMETERS
beta = 2.                           # coefficient controlling rain production
alpha2 = 6.                         # coefficient controlling rain removal
cc2 = 1.8                           # coefficient controlling suppressing of convection

''' FILTER PARAMETERS '''

dtmeasure = 0.089                   # duration of forecast step in non-dimensional time units [1h = 1/((L/U)/3600) if U in m/s and L in m]
n_ens = 20                          # number of ensemble members
TIMEOUT_PAR = n_ens*8	            # time to wait until all forecasts running in parallel are over
Nmeas = 96                          # number of total analysis steps
Nforec = 7			    # duration of the *dtmeasure*-long forecast launched at the end of each analysis step
NIAU = 1000			    # suppress injection of additional inflation with IAU for the first NIAU *dtmeasure* since assimilation 
IAU_dir = ''                        # directory name for experiments with different NIAU values
tn = 0.0                            # initial time
spin_up = 12                        # spin-up duration for plotting in *dtmeasure*: data during this period are discarded
tmax = (Nmeas+Nforec)*dtmeasure     # maximum duration of truth trajectory, from initial condition to end of last forecast in non-dim. time units
t_end_assim = Nmeas*dtmeasure       # duration of DA in non-dimensional time units
tmeasure = dtmeasure     
assim_time = np.linspace(tn,t_end_assim,Nmeas+1) # array of times of analysis steps
lead_times = [0,3,6]                # lead times for plotting
ass_freq='1h'                       # frequency of assimilation. This is only a string for file names and need to be set together with dtmeasure

### INITIAL CONDITION ###
sig_ic = [0.02,0.008,0.1,0.0]       # initial perturbation (st. dev) to generate initial conditions [sigma,u,v,r]
ic = init_cond_9                    # initial condition chosen from init_cond_ismodRSW

### Q MATRIX GENERATION ###
model_noise = [0.0,0.0,0.0,0.0]     # user-specified model noise to create model error covariance matrix Q 
Nhr = 1                             # Duration of model forecasts used to compute Q
Q_FUNC = 'Q_nhr()'                  # Q_predef(): user-specified error covariance matrix Q, Q_nhr(): error covariance matrix Q computed online
rMODNOISE = 1                       # binary parameter to control the presence of additive inflation in r (1=yes,0=no)
sigMODNOISE = 0                     # binary parameter to control the presence of additive inflation in sigma (1=yes,0=no)

### OBSERVING SYSTEM PARAMETERS ###
ob_noise = [0.01,0.0001,0.04,0.04,0.01]					# ob noise for [sat,T,sigu,sigv,sigr]
n_d = Neq*Nk_fc								# number of degrees of freedom (variable*grid points)
swathe = np.array([20,20,40,40,60,60,80,80])                		# width of each satellite Field of View (in Km)
sat_init_pos = np.array([0.1,0.64,0.17,0.75,0.35,0.88,0.41,0.9])	# position of satellites at first analysis step
sat_vel = np.array([0.2,-0.12,-0.25,0.18,0.15,-0.15,0.1,-0.1])		# velocity of satellites (x/h)
n_obs_sat = len(swathe)							# total number of satellite observations
obs_T_d = 1000								# observation spacing for temperature (in grid points)
obs_u_d = 20								# observation spacing for u (in grid points)
obs_v_d = 20								# observation spacing for v (in grid points)
obs_r_d = 20								# observation spacing for r (in grid points)
n_obs_T = Nk_fc // obs_T_d						# number of temperature observations
n_obs_u = Nk_fc // obs_u_d						# number of u observations
n_obs_v = Nk_fc // obs_v_d						# number of v observations
n_obs_r = Nk_fc // obs_r_d						# number of r observations
n_obs_grnd = n_obs_T+n_obs_u+n_obs_v+n_obs_r				# total number of ground observations
n_d_grnd = Neq*Nk_fc                  					# number of degrees of freedom for ground variables
n_obs = n_obs_grnd + n_obs_sat						# total number of observations
# Mask for satellite observations in the observation vector
sat_obs_mask = np.array(list(range(n_obs_sat)))
# Mask for temperature observations in the observation vector
sigT_obs_mask = np.array(list(range(n_obs_sat,n_obs_sat+n_obs_T)))
# Mask for u observations in the observation vector
sigu_obs_mask = np.array(list(range(n_obs_sat+n_obs_T,n_obs_sat+n_obs_T+n_obs_u)))
# Mask for v observations in the observation vector
sigv_obs_mask = np.array(list(range(n_obs_sat+n_obs_T+n_obs_u,n_obs_sat+n_obs_T+n_obs_u+n_obs_v)))
# Mask for r observations in the observation vector
sigr_obs_mask = np.array(list(range(n_obs_sat+n_obs_T+n_obs_u+n_obs_v,n_obs_sat+n_obs_T+n_obs_u+n_obs_v+n_obs_r)))

### STRING USED FOR PLOTTING (data denial experiments)
sim_title = 'control'
config_str = 'con'

### OUTER LOOP PARAMETERS (main_p.py)  ###
loc = [1.0] 			# list of localisation scale vales (must be floating point)
add_inf = [0.5]                 # list of additive inflation coefficient values (must be floating point)
rtpp = [0.5]                    # list of Relaxation To Prior Perturbation values (must be floating point)
rtps = [0.6]                    # list of Relaxation To Prior Spread values (must be floating point)
