########################################################################
### This script is a subroutine for batch-processing EnKF experiments  ###
########################################################################

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import os
import errno
import importlib.util
import sys
import multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings("error")
import shutil

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################
from f_ismodRSW import make_grid, time_step, ens_forecast
from f_enkf_ismodRSW import analysis_step_enkf_nl
from create_readme import create_readme
from isen_func import interp_sig2etab, M_int

def run_enkf(i, j, m, l, U_tr_array, Y_obs, dirname, config_file, h5_file_data):

    ################################################################
    # IMPORT PARAMETERS FROM CONFIGURATION FILE
    ################################################################

    spec = importlib.util.spec_from_file_location("config", sys.argv[1])
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    outdir = config.outdir
    cfl_fc = config.cfl_fc
    L = config.L
    A = config.A
    V = config.V
    Neq = config.Neq
    S0 = config.S0
    Nk_fc = config.Nk_fc
    Nk_tr = config.Nk_tr
    n_d = config.n_d
    dres = config.dres
    sig_c = config.sig_c
    sig_r = config.sig_r
    cc2 = config.cc2
    beta = config.beta
    alpha2 = config.alpha2
    Ro = config.Ro
    ob_noise = config.ob_noise    
    n_obs_sat = config.n_obs_sat
    n_obs_T = config.n_obs_T
    n_obs_u = config.n_obs_u
    n_obs_v = config.n_obs_v
    n_obs_r = config.n_obs_r
    sat_init_pos = config.sat_init_pos
    n_obs_grnd = config.n_obs_grnd
    obs_T_d = config.obs_T_d
    obs_u_d = config.obs_u_d
    obs_v_d = config.obs_v_d
    obs_r_d = config.obs_r_d
    n_obs = config.n_obs
    sat_vel = config.sat_vel
    sat_obs_mask = config.sat_obs_mask
    sigT_obs_mask = config.sigT_obs_mask
    sigu_obs_mask = config.sigu_obs_mask
    sigv_obs_mask = config.sigv_obs_mask
    sigr_obs_mask = config.sigr_obs_mask
    ic = config.ic
    sig_ic = config.sig_ic
    n_ens = config.n_ens
    TIMEOUT_PAR = config.TIMEOUT_PAR
    dtmeasure = config.dtmeasure
    assim_time = config.assim_time
    t_end_assim = config.t_end_assim
    Nmeas = config.Nmeas
    Nforec = config.Nforec
    NIAU = config.NIAU
    rtpp = config.rtpp
    rtps = config.rtps
    loc = config.loc
    add_inf = config.add_inf
    theta1 = config.theta1
    theta2 = config.theta2
    eta0 = config.eta0
    Z0 = config.Z0
    R = config.R
    k = config.k
    g = config.g 
    U_scale = config.U_scale
    U_relax = config.U_relax
    tau_rel = config.tau_rel

    print(' ')
    print('---------------------------------------------------')
    print('----------------- EXPERIMENT '+str(i+1)+str(j+1)+str(m+1)+str(l+1)+' ------------------')
    print('---------------------------------------------------')
    print(' ')

    print(np.shape(U_tr_array))

    pars_enda = [rtpp[m], rtps[l], loc[i], add_inf[j]]
    
    #################################################################
    # create directory for output
    #################################################################
    dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir, str(loc[i]), str(add_inf[j]), str(rtpp[m]), str(rtps[l]))
 
    # Delete and recreate the output directory.
    try:
        if os.path.isdir(dirn):
            shutil.rmtree(dirn)
        os.makedirs(dirn)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    ##################################################################
    # Load the model error covariance
    #################################################################
    Q = np.load(outdir+'/Qmatrix.npy') 

    ##################################################################
    # Mesh generation for forecasts
    ##################################################################

    fc_grid =  make_grid(Nk_fc,L) # forecast

    Kk_fc = fc_grid[0]
    x_fc = fc_grid[1]
    xc = fc_grid[2]

    ##################################################################    
    #  Apply initial conditions
    ##################################################################
    print(' ') 
    print('---------------------------------------------------') 
    print('---------      ICs: generate ensemble     ---------')
    print('---------------------------------------------------') 
    print(' ') 
    print('Initial condition =', str(ic), '(see <init_cond_modRSW.py> for deatils ... )')
    print(' ') 
    ### Forecast ic 
    U0_fc, B = ic(x_fc,Nk_fc,Neq,S0,L,A,V) # control IC to be perturbed

    U0ens = np.empty([Neq,Nk_fc,n_ens])

    print('Initial ensemble perurbations:')
    print('sig_ic = [sig_sig, sig_sigu, sig_sigv, sig_sigr] =', sig_ic)    

    # Generate initial ensemble
    for jj in range(0,Neq):
        for N in range(0,n_ens):
            # add sig_ic to EACH GRIDPOINT
            U0ens[jj,:,N] = U0_fc[jj,:] + sig_ic[jj]*np.random.randn(Nk_fc)

    # if hr < 0, set to zero:
    sigr = U0ens[3, :, :]
    sigr[sigr < 0.] = 0.
    U0ens[3, :, :] = sigr
    
    # if h < 0, set to epsilon:
    sig = U0ens[0, :, :]
    sig[sig < 0.] = 1e-3
    U0ens[0, :, :] = sig

    ### Convection and rain thresholds
    etab_c = interp_sig2etab(sig_c,h5_file_data)
    Mc = M_int(etab_c,0.,R,k,theta1,theta2,eta0,g,Z0,U_scale)
   
    ##################################################################
    #%%%-----        Define arrays for outputting data       ------%%%
    ##################################################################
    X_array = np.empty([n_d,n_ens,Nmeas])
    X_array.fill(np.nan)
    Xan_array = np.empty([n_d,n_ens,Nmeas])
    Xan_array.fill(np.nan)
    X_tr_array = np.empty([n_d,1,Nmeas])
    X_tr_array.fill(np.nan)
    X_forec = np.empty([n_d,n_ens,Nmeas,Nforec])
    OI = np.empty([Neq+2,Nmeas])
    OI.fill(np.nan)

    # create readme file of exp summary and save
    PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_enda, sig_ic, n_obs, NIAU, dres, sat_vel, n_obs_sat, n_obs_grnd, n_d, Neq, L, ob_noise, k, R, theta1, theta2, eta0, Z0, g, U_scale, sat_obs_mask, sigu_obs_mask, sigv_obs_mask, sigr_obs_mask, sig_c, sig_r, sat_init_pos, obs_T_d, obs_u_d, obs_v_d, obs_r_d, n_obs_T, n_obs_u, n_obs_v, n_obs_r, sigT_obs_mask]
    create_readme(dirn, config_file, i, j, m, l)
    
    ##################################################################
    #  Integrate ensembles forward in time until obs. is available   #
    ##################################################################
    print(' ')
    print('-------------------------------------------------')
    print('------ CYCLED FORECAST-ASSIMILATION SYSTEM ------')
    print('-------------------------------------------------')
    print('--------- ENSEMBLE FORECASTS + EnKF--------------')
    print('-------------------------------------------------')
    print(' ')
    
    # Initialise...
    U = U0ens
    index = 0 # to step through assim_time
    tmeasure = dtmeasure    
    
    U_rel = U_relax(Neq,Nk_fc,L,V,xc,U_tr_array[:,0::dres,0])
 
    while tmeasure-dtmeasure < t_end_assim and index < Nmeas:
        
        print(np.shape(U))

        try:

            if index==0:
               
                print(' ')
                print('----------------------------------------------')
                print('------------ FORECAST STEP: START ------------')
                print('----------------------------------------------')
                print(' ')
        
                num_cores_use = os.cpu_count()
           
                print('Starting ensemble integrations from time =', assim_time[index],' to',assim_time[index+1]) 
                print('Number of cores used:', num_cores_use)
                print(' *** Started: ', str(datetime.now()))
                
                print(np.shape(U))
        
                ### ADDITIVE INFLATION (moved to precede the forecast) ###
                q = add_inf[j] * np.random.multivariate_normal(np.zeros(n_d), Q, n_ens)
                q_ave = np.mean(q,axis=0)
                q = q - q_ave
                q = q.T
        
                pool = mp.Pool(processes=num_cores_use)
                mp_out = [pool.apply_async(ens_forecast, args=(N, U, U_rel, tau_rel, q, Neq, Nk_fc, Kk_fc, cfl_fc, assim_time, index, tmeasure, dtmeasure, Mc, sig_c, sig_r, cc2, beta, alpha2, Ro, R, k, theta1, theta2, eta0, g, Z0, U_scale, h5_file_data)) for N in range(0,n_ens)]
                U = [p.get(timeout=60) for p in mp_out]

                pool.close()
                pool.join()

                print(' All ensembles integrated forward from time =', assim_time[index],' to',assim_time[index+1])
                print(' *** Ended: ', str(datetime.now()))
                print(np.shape(U))

                U=np.swapaxes(U,0,1)
                U=np.swapaxes(U,1,2)

                print(np.shape(U))

                dU = np.copy(U)
                dU = U - np.repeat(U_tr_array[:,0::dres,index+1], n_ens).reshape([Neq, Nk_fc, n_ens])
                print("Forecast", dU[0].min(), dU[0].max(), dU[1].min(), dU[1].max(), dU[2].min(), dU[2].max(), dU[3].min(), dU[3].max())
                print(' ')
                print('----------------------------------------------')
                print('------------- FORECAST STEP: END -------------')
                print('----------------------------------------------')
                print(' ')

            ##################################################################
            #  calculate analysis at observing time then integrate forward  #
            ##################################################################

            U_an, U_fc, X_array[:,:,index], X_tr_array[:,0,index], Xan_array[:,:,index], OI[:,index] = analysis_step_enkf_nl(U, U_tr_array, Y_obs, tmeasure, dtmeasure, index, PARS, h5_file_data, dirn)
        
            U = U_an # update U with analysis ensembles for next integration
            dU = U_an - np.repeat(U_tr_array[:,0::dres,index+1], n_ens).reshape([Neq, Nk_fc, n_ens])
            
            #######################################
            #  generate a *Nforec*-long forecast # 
            #######################################

            print(' Long-range forecast starting... ')
                
            tforec = tmeasure
            tendforec = tforec + Nforec*dtmeasure
            forec_time = np.linspace(tforec,tendforec,Nforec+1)
            forec_T = 1
            U_forec = np.copy(U_an)

            X_forec[:,:,index,0] = np.copy(X_array[:,:,index])

            while tforec < tendforec and forec_T < Nforec: 
                
                ### ADDITIVE INFLATION (moved to precede the forecast) ###
                q = add_inf[j] * np.random.multivariate_normal(np.zeros(n_d), Q, n_ens)
                q_ave = np.mean(q,axis=0)
                q = q - q_ave
                q = q.T
  
                if forec_T > NIAU: q[:,:] = 0.0
                
                pool = mp.Pool(processes=num_cores_use)

                mp_out = [pool.apply_async(ens_forecast, args=(N, U_forec, U_rel, tau_rel, q, Neq, Nk_fc, Kk_fc, cfl_fc, forec_time, forec_T-1, tforec+dtmeasure, dtmeasure, Mc, sig_c, sig_r, cc2, beta, alpha2, Ro, R, k, theta1, theta2, eta0, g, Z0, U_scale,h5_file_data)) for N in range(0,n_ens)]
                U_forec = [p.get(timeout=TIMEOUT_PAR) for p in mp_out]

                pool.close()
                pool.join()

                U_forec = np.swapaxes(U_forec,0,1)
                U_forec = np.swapaxes(U_forec,1,2)

                if forec_T==1: U = np.copy(U_forec)

                U_forec_tmp = np.copy(U_forec)
                U_forec_tmp[1:,:,:] = U_forec_tmp[1:,:,:]/U_forec_tmp[0,:,:]
 
                for N in range(n_ens):
                    X_forec[:,N,index,forec_T] = U_forec_tmp[:,:,N].flatten()

                print(' All ensembles integrated forward from time =', round(tforec,3) ,' to', round(tforec+dtmeasure,3))
                
                tforec = tforec+dtmeasure
                forec_T = forec_T + 1

            # on to next assim_time
            index = index + 1
            tmeasure = tmeasure + dtmeasure

        except (RuntimeWarning, mp.TimeoutError) as err:
            pool.terminate()
            pool.join()
            print(err)
            print('-------------- Forecast failed! --------------')        
            print(' ')
            print('----------------------------------------------')
            print('------------- FORECAST STEP: END -------------')
            print('----------------------------------------------')
            print(' ')

            tmeasure = t_end_assim + dtmeasure

        except IndexError as err:

            print('-------------- Analysis failed! --------------')
            print('----------------------------------------------')
            print(' ')

            tmeasure = t_end_assim + dtmeasure


    ##################################################################

    # create readme file and save
    create_readme(dirn, config_file, i, j, m, l)
    np.save(str(dirn+'/B'),B)
    np.save(str(dirn+'/X_array'),X_array)
    np.save(str(dirn+'/X_tr_array'),X_tr_array)
    np.save(str(dirn+'/X_forec'),X_forec)
    np.save(str(dirn+'/Xan_array'),Xan_array)
    np.save(str(dirn+'/OI'),OI)

    print(' *** Data saved in :', dirn)
    print(' ')

    # print summary to terminal as well
    print(' ') 
    print('---------------------------------------') 
    print('--------- END OF ASSIMILATION ---------') 
    print('---------------------------------------') 
    print(' ')   
    print(' -------------- SUMMARY: ------------- ')  
    print(' ') 
    print('Dynamics:')
    print('Ro =', Ro)  
    print('(H_0 , H_c , H_r) =', [S0, sig_c, sig_r]) 
    print('(alpha, beta, c2) = ', [alpha2, beta, cc2])
    print('cfl = ', cfl_fc)
    print('Initial condition =', str(ic))
    print(' ') 
    print('Assimilation:')
    print('Forecast resolution (number of gridcells) =', Nk_fc)
    print('Truth resolution (number of gridcells) =', Nk_tr)   
    if Nk_fc == Nk_tr: # perfect model
        print('>>> perfect model scenario')
    else:
        print('>>> imperfect model scenario') 
    print(' ')  
    print('Number of ensembles =', n_ens)  
    #print('Observation density: observe every', pars_ob[0], 'gridcells...')
    #print('i.e., total no. of obs. =', Nk_fc*Neq/pars_ob[0])
    print('Observation noise =', ob_noise)  
    print('RTPP (ensemble) factor =', pars_enda[0])
    print('RTPS (ensemble) factor =', pars_enda[1])
    print('Additive inflation factor =', pars_enda[3])
    print('Localisation lengthscale =', pars_enda[2])
    print(' ')   
    print(' ----------- END OF SUMMARY: ---------- ')  
    print(' ')  

    ##################################################################
    #                       END OF PROGRAM                           #
    ################################################################## 
