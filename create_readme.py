#######################################################################
# Create readme.txt file for summarising each experiment, output saved in 
# <dirn> to accompany outputted data from main run script and EnKF subroutine.
#######################################################################

#from parameters import *
from datetime import datetime
import os
import importlib

def create_readme(dirn, config_file, i, j, m, l):
    '''
    INPUT:
    > dirn = directory path
    > PARS = [Nk_fc, Nk_tr, n_ens, assim_time, pars_enda, sig_ic, n_obs]
    where pars = [obs_dens, rtpp, rtps, ob_noise]  
    > ic = initial condition
        
    OUTPUT:
    > fname.txt file saved in dirn
    '''   

    ################################################################
    # IMPORT PARAMETERS FROM CONFIGURATION FILE
    ################################################################

    spec = importlib.util.spec_from_file_location("config", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    Nk_tr = config.Nk_tr
    Nk_fc = config.Nk_fc
    L = config.L
    Neq = config.Neq
    S0 = config.S0
    A = config.A
    V = config.V
    Ro = config.Ro
    sig_c = config.sig_c
    sig_r = config.sig_r
    alpha2 = config.alpha2
    beta = config.beta
    cc2 = config.cc2
    g = config.g
    cfl_fc = config.cfl_fc
    cfl_tr = config.cfl_tr
    n_ens = config.n_ens
    ic = config.ic
    sig_ic = config.sig_ic
    Nmeas = config.Nmeas
    Nforec = config.Nforec
    NIAU = config.NIAU
    n_obs = config.n_obs
    ob_noise = config.ob_noise
    assim_time = config.assim_time
    loc = config.loc
    add_inf = config.add_inf
    rtpp = config.rtpp
    rtps = config.rtps 
        
    fname = str(dirn+'/readme.txt')

    f = open(fname,'w')
    print(' ------------- FILENAME ------------- ', file=f) 
    print(fname, file=f)   
    print(' ', file=f)   
    print('Created: ', str(datetime.now()), file=f)   
    print(' ', file=f)        
    print('Output <.npy> saved to directory:', file=f) 
    print(str(dirn), file=f)   
    print(' ', file=f)   
    print(' -------------- SUMMARY: ------------- ', file=f)  
    print(' ', file=f) 
    print('Dynamics:', file=f)
    print(' ', file=f) 
    print('Ro =', Ro, file=f)  
    print('(H_0 , H_c , H_r) =', [S0, sig_c, sig_r], file=f) 
    print('(alpha, beta, c2) = ', [alpha2, beta, cc2], file=f)
    print('(cfl_fc, cfl_tr) = ', [cfl_fc, cfl_tr], file=f)
    print('Initial condition =', str(ic), file=f)
    print('IC noise for initial ens. generation: ', sig_ic, file=f)
    print(' ', file=f) 
    print('Assimilation:', file=f)
    print(' ', file=f) 
    print('Forecast resolution (number of gridcells) =', Nk_fc, file=f)
    print('Truth resolution (number of gridcells) =', Nk_tr, file=f)   
    if Nk_fc == Nk_tr: # perfect model
        print('            >>> perfect model scenario', file=f)
    else: # imperfect model
        print('            >>> imperfect model scenario', file=f) 
    print(' ')  
    print('Number of ensembles =', n_ens, file=f)  
#    print >>f, 'Assimilation times  =', PARS[3][1:]
#    print >>f, 'Observation density: observe every', PARS[4][0], 'gridcells...'
    print('i.e., total no. of obs. =', n_obs, file=f)
    print('Observation noise =', ob_noise, file=f)
    if rtpp[m] != 1.0: # inflate the ensemble
        print('RTPP factor =', rtpp[m], file=f)
    else: # no inflation
        print('No RTPP applied', file=f)
    if rtps[l] != 1.0: # inflate the ensemble
        print('RTPS factor =', rtps[l], file=f)
    else: # no inflation
        print('No RTPS applied', file=f)
    print('Additive inflation factor =', add_inf[j], file=f)
    print('Localisation lengthscale =', loc[i], file=f)
    print(' ', file=f)   
    print('Duration of the Incremental Analysis Update into the forecast (in number of cycles) = ', NIAU, file=f)
    print(' ----------- END OF SUMMARY: ---------- ', file=f)  
    print(' ', file=f)  
    f.close()
