##################################################################
### This script calculate error stats for saved data
##################################################################

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import sys

##################################################################
# CUSTOM FUNCTIONS AND MODULES REQUIRED
##################################################################
from crps_calc_fun import crps_calc

##################################################################
### Function that calculates the stats for the analysis

def ave_stats_an(Nk_fc,Neq,n_d,n_ens,Nmeas,spin_up,indices,outdir,loc,add_inf,rtpp,rtps,X_tr):
    
    # LOAD DATA FROM GIVEN DIRECTORY
    dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir,
                                                     str(loc[indices[0]]),
                                                     str(add_inf[indices[1]]),
                                                     str(rtpp[indices[2]]),
                                                     str(rtps[indices[3]]))
    
    if os.path.exists(dirn):
        print(' ')
        print('Path: ')
        print(dirn)
        print(' exists... calculating stats...')
        print(' ')
    
        # LOAD DATA FROM GIVEN DIRECTORY
        Xan = np.load(str(dirn+'/Xan_array.npy')) # an ensembles
        OI = np.load(str(dirn+'/OI.npy')) # obs impact
    
        # masks for locating model variables in state vector
        sig_mask = list(range(0,Nk_fc))
        sigu_mask = list(range(Nk_fc,2*Nk_fc))
        if(Neq==3):
            sigr_mask = list(range(2*Nk_fc,3*Nk_fc))
        if(Neq==4):
            sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
            sigr_mask = list(range(3*Nk_fc,4*Nk_fc))  

        # create mask to compute averages 
        nz_index = np.where(OI[0,spin_up:]) - np.repeat(1,len(np.where(OI[0,spin_up:])))
        
        # create time vector to run over assimilation cycles
        time_vec = list(range(0,Nmeas))

        spr_an_ave = np.empty(Neq+1)
        err_an_ave = np.empty(Neq+1)
        rmse_an_ave = np.empty(Neq+1)
        crps_an_ave = np.empty(Neq+1)
        sprrmse_ratio_an = np.empty(Neq+1)
        OI_ave = np.empty(Neq+2)

        if (np.any(np.isnan(OI[0,:]))):
            print('Runs crashed before Tmax...')

            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
        
        else:
            ##################################################################
            print('Runs completed... ')
            # for means and deviations
            Xanbar = np.empty(np.shape(Xan))
            Xandev = np.empty(np.shape(Xan))
            Xandev_tr = np.empty(np.shape(Xan))

            # for errs as at each assim time
            rmse_an = np.empty((Neq,len(time_vec)))
            spr_an = np.empty((Neq,len(time_vec)))
            ame_an = np.empty((Neq,len(time_vec)))
            crps_an = np.empty((Neq,len(time_vec)))

            print(' *** Calculating errors from ', dirn)

            for T in time_vec:
                
                plt.clf() # clear figs from previous loop
                
                Xanbar[:,:,T] = np.repeat(Xan[:,:,T].mean(axis=1), n_ens).reshape(n_d, n_ens)
                Xandev[:,:,T] = Xan[:,:,T] - Xanbar[:,:,T] # an deviations from mean
                Xandev_tr[:,:,T] = Xan[:,:,T] - X_tr[:,:,T] # an deviations from truth

                ##################################################################
                ###                       ERRORS                              ####
                ##################################################################

                # ANALYSIS: mean error
                an_err = Xanbar[:,0,T] - X_tr[:,0,T] # an_err = analysis ens. mean - truth
                an_err2 = an_err**2
                
                # ANALYSIS: cov matrix for spread...
                Pa = np.dot(Xandev[:,:,T],np.transpose(Xandev[:,:,T]))
                Pa = Pa/(n_ens - 1) # analysis covariance matrix
                var_an = np.diag(Pa)
                
                # ... and rmse
                Pa_tr = np.dot(Xandev_tr[:,:,T],np.transpose(Xandev_tr[:,:,T]))
                Pa_tr = Pa_tr/(n_ens - 1) # fc covariance matrix w.r.t truth
                var_ant = np.diag(Pa_tr)
                
                ##################################################################
                ###                       CRPS                                ####
                ##################################################################
                
                CRPS_an = np.empty((Neq,Nk_fc))

                for ii in sig_mask:
                    CRPS_an[0,ii] = crps_calc(Xan[ii,:,T],X_tr[ii,0,T])
                    CRPS_an[1,ii] = crps_calc(Xan[ii+Nk_fc,:,T],X_tr[ii+Nk_fc,0,T])
                    CRPS_an[2,ii] = crps_calc(Xan[ii+2*Nk_fc,:,T],X_tr[ii+2*Nk_fc,0,T])
                    if(Neq==4): CRPS_an[3,ii] = crps_calc(Xan[ii+3*Nk_fc,:,T],X_tr[ii+3*Nk_fc,0,T])

                #################################################################
              

                # domain-averaged errors
                ame_an[0,T] = np.mean(np.absolute(an_err[sig_mask]))
                spr_an[0,T] = np.sqrt(np.mean(var_an[sig_mask]))
                rmse_an[0,T] = np.sqrt(np.mean(an_err2[sig_mask]))
                crps_an[0,T] = np.mean(CRPS_an[0,:])

                ame_an[1,T] = np.mean(np.absolute(an_err[sigu_mask]))
                spr_an[1,T] = np.sqrt(np.mean(var_an[sigu_mask]))
                rmse_an[1,T] = np.sqrt(np.mean(an_err2[sigu_mask]))
                crps_an[1,T] = np.mean(CRPS_an[1,:])

                if(Neq==3):
                    ame_an[2,T] = np.mean(np.absolute(an_err[sigr_mask]))
                    spr_an[2,T] = np.sqrt(np.mean(var_an[sigr_mask]))
                    rmse_an[2,T] = np.sqrt(np.mean(an_err2[sigr_mask]))
                    crps_an[2,T] = np.mean(CRPS_an[2,:])
                if(Neq==4):
                    ame_an[2,T] = np.mean(np.absolute(an_err[sigu_mask]))
                    spr_an[2,T] = np.sqrt(np.mean(var_an[sigu_mask]))
                    rmse_an[2,T] = np.sqrt(np.mean(an_err2[sigu_mask]))
                    crps_an[2,T] = np.mean(CRPS_an[2,:])
                    ame_an[3,T] = np.mean(np.absolute(an_err[sigr_mask]))
                    spr_an[3,T] = np.sqrt(np.mean(var_an[sigr_mask]))
                    rmse_an[3,T] = np.sqrt(np.mean(an_err2[sigr_mask]))
                    crps_an[3,T] = np.mean(CRPS_an[3,:])
        

           ###########################################################################
 
            spr_an_ave[0:Neq] = np.ravel(spr_an[:,nz_index].mean(axis=-1))
            err_an_ave[0:Neq] = np.ravel(ame_an[:,nz_index].mean(axis=-1))
            rmse_an_ave[0:Neq] = np.ravel(rmse_an[:,nz_index].mean(axis=-1))
            crps_an_ave[0:Neq] = np.ravel(crps_an[:,nz_index].mean(axis=-1))
            OI_ave = np.flip(np.ravel(100*OI[:,nz_index].mean(axis=-1)),0)
 
            if(Neq==3):
                spr_an_ave[Neq] = np.mean((spr_an_ave[0],spr_an_ave[1],spr_an_ave[2]))
                err_an_ave[Neq] = np.mean((err_an_ave[0],err_an_ave[1],err_an_ave[2]))
                rmse_an_ave[Neq] = np.mean((rmse_an_ave[0],rmse_an_ave[1],rmse_an_ave[2]))

            if(Neq==4):
                spr_an_ave[Neq] = np.mean((spr_an_ave[0],spr_an_ave[1],spr_an_ave[3]))
                err_an_ave[Neq] = np.mean((err_an_ave[0],err_an_ave[1],err_an_ave[3]))
                rmse_an_ave[Neq] = np.mean((rmse_an_ave[0],rmse_an_ave[1],rmse_an_ave[3]))
            
            crps_an_ave[Neq] = np.mean((crps_an_ave[0],crps_an_ave[1],crps_an_ave[3]))
            sprrmse_ratio_an = spr_an_ave/rmse_an_ave

            print('spr_an ave. =', spr_an_ave)
            print('err_an ave. =', err_an_ave)
            print('rmse_an ave. =', rmse_an_ave)
            print('crps_an_ave. =', crps_an_ave)
            print('OI ave. =', OI_ave)

            return spr_an_ave, err_an_ave, rmse_an_ave, crps_an_ave, sprrmse_ratio_an, OI_ave

    else:
        print(' ')
        print(' Path:')
        print(dirn)
        print('does not exist.. moving on to next one...')
        print(' ')

        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

##################################################################
### Function that calculates the stats for the forecast at given lead time
    
def ave_stats_fc(Nk_fc,Neq,n_d,n_ens,Nmeas,spin_up,indices,outdir,loc,add_inf,rtpp,rtps,lead_time,X_tr):
    
    # LOAD DATA FROM GIVEN DIRECTORY
    dirn = '{}/loc_{}_add_{}_rtpp_{}_rtps_{}'.format(outdir,
                                                     str(loc[indices[0]]),
                                                     str(add_inf[indices[1]]),
                                                     str(rtpp[indices[2]]),
                                                     str(rtps[indices[3]]))
    if os.path.exists(dirn):
        print(' ')
        print('Path: ')
        print(dirn)
        print(' exists... calculating stats...')
        print(' ')
        
        # LOAD DATA FROM GIVEN DIRECTORY
        Xforec = np.load(str(dirn+'/X_forec.npy')) # long-range forecast
        OI = np.load(str(dirn+'/OI.npy')) # obs impact

        # masks for locating model variables in state vector
        sig_mask = list(range(0,Nk_fc))
        sigu_mask = list(range(Nk_fc,2*Nk_fc))
        if(Neq==3):
            sigr_mask = list(range(2*Nk_fc,3*Nk_fc))
        if(Neq==4):   
            sigv_mask = list(range(2*Nk_fc,3*Nk_fc))
            sigr_mask = list(range(3*Nk_fc,4*Nk_fc))

        # create mask to compute averages 
        nz_index = np.where(OI[0,spin_up:]) - np.repeat(1,len(np.where(OI[0,spin_up:])))
        
        # create time vector to run over assimilation cycles
        time_vec = list(range(0,Nmeas))

        spr_fc_ave = np.empty(Neq+1)
        err_fc_ave = np.empty(Neq+1)
        rmse_fc_ave = np.empty(Neq+1)
        crps_fc_ave = np.empty(Neq+1)
        sprrmse_ratio_fc = np.empty(Neq+1)

        if (np.any(np.isnan(OI[0,:]))):
            print('Runs crashed before Tmax...')

            return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    
        else:
            ##################################################################
            print('Runs completed... ')
            # for means and deviations
            Xforbar = np.empty(np.shape(Xforec))
            Xfordev = np.empty(np.shape(Xforec)) 
            Xfordev_tr = np.empty(np.shape(Xforec))

            # for errs as at each assim time
            rmse_fc = np.empty((Neq,len(time_vec)))
            spr_fc = np.empty((Neq,len(time_vec)))
            ame_fc = np.empty((Neq,len(time_vec)))
            crps_fc = np.empty((Neq,len(time_vec)))

            print(' *** Calculating errors from ', dirn)

            for T in time_vec:
                
                plt.clf() # clear figs from previous loop
                
                ### CALCULATING ERRORS AT DIFFERENT LEAD TIMES ###
                Xforbar[:,:,T,lead_time] = np.repeat(Xforec[:,:,T,lead_time].mean(axis=1), n_ens).reshape(n_d, n_ens)
                Xfordev[:,:,T,lead_time] = Xforec[:,:,T,lead_time] - Xforbar[:,:,T,lead_time]
                Xfordev_tr[:,:,T,lead_time] = Xforec[:,:,T,lead_time] - X_tr[:,:,T+lead_time]
               
                # LONG-RANGE FORECAST: mean error
                for_err = Xforbar[:,0,T,lead_time] - X_tr[:,0,T+lead_time]
                for_err2 = for_err**2
                
                # LONG-RANGE FORECAST: cov matrix for spread...
                Pflong = np.dot(Xfordev[:,:,T,lead_time],np.transpose(Xfordev[:,:,T,lead_time]))
                Pflong = Pflong/(n_ens - 1)
                var_longfc = np.diag(Pflong)

                # ... and rmse
                Pflong_tr = np.dot(Xfordev_tr[:,:,T,lead_time],np.transpose(Xfordev_tr[:,:,T,lead_time]))
                Pflong_tr = Pflong_tr/(n_ens - 1)
                var_longfct = np.diag(Pflong_tr)

                ##################################################################
                ###                       CRPS                                ####
                ##################################################################

                CRPS_fc = np.empty((Neq,Nk_fc))

                for ii in sig_mask:
                    CRPS_fc[0,ii] = crps_calc(Xforec[ii,:,T,lead_time],X_tr[ii,0,T+lead_time])
                    CRPS_fc[1,ii] = crps_calc(Xforec[ii+Nk_fc,:,T,lead_time],X_tr[ii+Nk_fc,0,T+lead_time])
                    CRPS_fc[2,ii] = crps_calc(Xforec[ii+2*Nk_fc,:,T,lead_time],X_tr[ii+2*Nk_fc,0,T+lead_time])
                    if(Neq==4): CRPS_fc[3,ii] = crps_calc(Xforec[ii+3*Nk_fc,:,T,lead_time],X_tr[ii+3*Nk_fc,0,T+lead_time])

                #################################################################
              

                # domain-averaged errors
                ame_fc[0,T] = np.mean(np.absolute(for_err[sig_mask]))
                spr_fc[0,T] = np.sqrt(np.mean(var_longfc[sig_mask]))
                rmse_fc[0,T] = np.sqrt(np.mean(for_err2[sig_mask]))
                crps_fc[0,T] = np.mean(CRPS_fc[0,:])

                ame_fc[1,T] = np.mean(np.absolute(for_err[sigu_mask]))
                spr_fc[1,T] = np.sqrt(np.mean(var_longfc[sigu_mask]))
                rmse_fc[1,T] = np.sqrt(np.mean(for_err2[sigu_mask]))
                crps_fc[1,T] = np.mean(CRPS_fc[1,:])

                if(Neq==3):
                    ame_fc[2,T] = np.mean(np.absolute(for_err[sigr_mask]))
                    spr_fc[2,T] = np.sqrt(np.mean(var_longfc[sigr_mask]))
                    rmse_fc[2,T] = np.sqrt(np.mean(for_err2[sigr_mask]))
                    crps_fc[2,T] = np.mean(CRPS_fc[2,:])

                if(Neq==4):
                    ame_fc[2,T] = np.mean(np.absolute(for_err[sigv_mask]))
                    spr_fc[2,T] = np.sqrt(np.mean(var_longfc[sigv_mask]))
                    rmse_fc[2,T] = np.sqrt(np.mean(for_err2[sigv_mask]))
                    crps_fc[2,T] = np.mean(CRPS_fc[2,:])
                    ame_fc[3,T] = np.mean(np.absolute(for_err[sigr_mask]))
                    spr_fc[3,T] = np.sqrt(np.mean(var_longfc[sigr_mask]))
                    rmse_fc[3,T] = np.sqrt(np.mean(for_err2[sigr_mask]))
                    crps_fc[3,T] = np.mean(CRPS_fc[3,:])
        
           ###########################################################################

            spr_fc_ave[0:Neq] = np.ravel(spr_fc[:,nz_index].mean(axis=-1))
            err_fc_ave[0:Neq] = np.ravel(ame_fc[:,nz_index].mean(axis=-1))
            rmse_fc_ave[0:Neq] = np.ravel(rmse_fc[:,nz_index].mean(axis=-1))
            crps_fc_ave[0:Neq] = np.ravel(crps_fc[:,nz_index].mean(axis=-1))
 
            if(Neq==3):
                spr_fc_ave[Neq] = np.mean((spr_fc_ave[0],spr_fc_ave[1],spr_fc_ave[2]))
                err_fc_ave[Neq] = np.mean((err_fc_ave[0],err_fc_ave[1],err_fc_ave[2]))
                rmse_fc_ave[Neq] = np.mean((rmse_fc_ave[0],rmse_fc_ave[1],rmse_fc_ave[2]))
            if(Neq==4):            
                spr_fc_ave[Neq] = np.mean((spr_fc_ave[0],spr_fc_ave[1],spr_fc_ave[3]))
                err_fc_ave[Neq] = np.mean((err_fc_ave[0],err_fc_ave[1],err_fc_ave[3]))
                rmse_fc_ave[Neq] = np.mean((rmse_fc_ave[0],rmse_fc_ave[1],rmse_fc_ave[3]))


            crps_fc_ave[Neq] = np.mean((crps_fc_ave[0],crps_fc_ave[1],crps_fc_ave[3]))
            sprrmse_ratio_fc = spr_fc_ave/rmse_fc_ave            

            print('spr_fc ave. =', spr_fc_ave)
            print('err_fc ave. =', err_fc_ave)
            print('rmse_fc ave. =', rmse_fc_ave)
            print('crps_fc ave. =', crps_fc_ave)

            return spr_fc_ave, err_fc_ave, rmse_fc_ave, crps_fc_ave, sprrmse_ratio_fc

    else:
        print(' ')
        print(' Path:')
        print(dirn)
        print('does not exist.. moving on to next one...')
        print(' ')

        return float('nan'), float('nan'), float('nan'),  float('nan'), float('nan')
