##################################################################
# Summary diagnostics of idealised enkf experiments
# inc. summary plots a la Poterjoy and Zhang
##################################################################

'''
    Each directory has i*j*k experiments with different parameter combinations. This script produces summary plots for comparison. Assume only one RTPP value
when plotting.
    
    '''

## generic modules
import os
import errno
import itertools
import sys
import importlib.util
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from mpl_toolkits.axes_grid1 import AxesGrid

## custom modules
from analysis_diag_stats import ave_stats_an, ave_stats_fc

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir
loc = config.loc
add_inf = config.add_inf
rtpp = config.rtpp
rtps = config.rtps
Neq = config.Neq
dres = config.dres
Nk_fc = config.Nk_fc
Nmeas = config.Nmeas
Nforec = config.Nforec
n_d = config.n_d
lead_times = config.lead_times
n_ens = config.n_ens
spin_up = config.spin_up
ass_freq = config.ass_freq

if Neq==3: var=['sig','u','r','all']
if Neq==4: 
    var=['sig','u','v','r','all']
    var_oi = ['sat','T','u','v','r','all']

# lOAD TRUTH TRAJECTORY
U_tr = np.load(str(outdir+'/U_tr_array_2xres_'+ass_freq+'.npy'))

# Manipulate U_tr to get X_tr
U_tr_tmp = np.empty((Neq,Nk_fc,Nforec+Nmeas))
for i in range(0,Nk_fc):
    U_tr_tmp[:,i,:] = U_tr[:,i*dres,1:]
U_tr_tmp[1:,:,:] = U_tr_tmp[1:,:,:]/U_tr_tmp[0,:,:]

X_tr = np.empty((n_d,1,Nmeas+Nforec))
for k in range(0,Nmeas+Nforec):
    X_tr[:,0,k] = U_tr_tmp[:,:,k].flatten()

# generic plotting subroutine
def summary_plot(x, var_name, graph_title, axis_labels, values, minval, maxval, filnam,mymap):
    '''Plot the three-dimensional array x as a sequence of slices, highlighting
    the minimum cell with a blue dot and masked values with green cells.'''
    indmin = np.unravel_index(np.argmin(x), x.shape)
    if (var_name == 'sprrmse_ratio'): 
        indint = np.dstack(np.where((x>0.8) & (x<1.2)))
        print(indint)
    if (var_name == 'rmse'):
        rtps = np.array((4,5,6,5,6,4,5,7,4,7,6,6))
        gama = np.array((3,4,0,2,2,3,3,3,4,1,2,5))
        lloc = np.array((0,0,1,1,1,1,2,2,2,3,3,3))
        list_rmse = np.dstack((rtps,gama,lloc))
        print(list_rmse)
    if (var_name == 'crps'):
        rtps = np.array((4,5,6,5,6,4,5,7,4,7,6,6))
        gama = np.array((3,4,0,2,2,3,3,3,4,1,2,5))
        lloc = np.array((0,0,1,1,1,1,2,2,2,3,3,3))
        list_crps = np.dstack((rtps,gama,lloc))
        print(list_crps)
    if len(indmin) != 3:
        raise ValueError('summary_plot requires a rank 3 data array')
    if len(axis_labels) != 3:
        raise ValueError('summary plot requires a rank 3 label array')
    if len(values) != 3:
        raise ValueError('summary plot requires a rank 3 tick array')

    x_dim = np.shape(x)
    n_plots = x_dim[2]
#    if n_plots % 2 != 0:
#        raise ValueError('an even number of elements is required in the '
#                         'third rank of the data array')
   # Produce pages of at most 2 x 2 plots#
    inf = int(np.floor(np.sqrt(n_plots)))
    sup = int(np.ceil(np.sqrt(n_plots)))
    a = int(inf*inf)
    b = int(inf*sup)
    if n_plots>b:
       n_rows = sup
       n_cols = sup
    elif n_plots>a:
       n_rows = sup
       n_cols = inf
    else:
       n_rows = inf
       n_cols = inf
    fig = plt.figure(figsize=(7.,6.))
    print(n_rows,n_cols,n_plots)
    fig.suptitle('{}'.format(graph_title, x[indmin]),
                 fontsize='x-large')
    axlist = AxesGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=(0.25,0.3),
                      cbar_location='right', cbar_mode='single',
                      share_all=True)
    # Produce pages of at most 2 x 2 plots
#    n_rows = int(np.ceil(0.5 * n_plots))
#    fig = plt.figure()
#    fig.suptitle('{} (minimum = {:.3f})'.format(graph_title, x[indmin]),
#                 fontsize='large')
#    axlist = AxesGrid(fig, 111, nrows_ncols=(n_rows, 2), axes_pad=0.25,
#                      cbar_location='right', cbar_mode='single',
#                      share_all=True)

    for i, ax in enumerate(axlist):
        #mymap = cm.YlOrBr
        mymap.set_bad('g', 1.0)
        im = ax.pcolormesh(x[:, :, i], cmap=mymap, vmin=minval, vmax=maxval)

        # Shared legend.
        axlist.cbar_axes[0].colorbar(im)
        axlist.cbar_axes[0].tick_params(labelsize='large')

        # Avoid unwanted blank rows and/or columns.
        ax.set_adjustable('box')

        # Annotate.
        ax.set_title('{} = {}'.format(axis_labels[2], values[2][i]),
                     fontsize='large')
        ax.set_xticks(np.arange(0.5, x_dim[1], 1))
        ax.set_xticklabels(values[1], rotation=90, fontsize='large')
        ax.set_xlabel(axis_labels[1],fontsize='large')
        ax.set_yticks(np.arange(0.5, x_dim[0], 1))
        ax.set_yticklabels(['{x:.2f}'.format(x=a) for a in values[0]], fontsize='large')
        ax.set_ylabel(axis_labels[0],fontsize='large')

	# Highlight the cells with 0.8 < SPR/RMSE < 1.2
        if var_name == 'sprrmse_ratio' and np.any(indint[:,:,2]==i):
            for j in indint[np.where(indint[:,:,2]==i)]:
                ax.add_patch(patches.Rectangle((j[1],j[0]),1.,1.,edgecolor='r',fill=False))

        # Highlight the cells with low values of RMSE and CRPS
        if var_name == 'rmse' and np.any(list_rmse[:,:,2]==i):
            for j in list_rmse[np.where(list_rmse[:,:,2]==i)]:
                ax.add_patch(patches.Rectangle((j[1],j[0]),1.,1.,edgecolor='black',fill=False))
        if var_name == 'crps' and np.any(list_crps[:,:,2]==i):
            for j in list_crps[np.where(list_crps[:,:,2]==i)]:
                ax.add_patch(patches.Rectangle((j[1],j[0]),1.,1.,edgecolor='black',fill=False))

        # Highlight the cell with the minimum value.
#        if i == indmin[2]:
#            ax.plot(indmin[1]+0.5, indmin[0]+0.5, color='blue', marker='o',
#                    markersize=10)
    plt.savefig(filnam)

##################################################################

# LOAD DATA FROM GIVEN DIRECTORY
figsdir = str(outdir+'/figs')

#check if dir exists, if not make it
try:
    os.makedirs(figsdir)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

##################################################################

spr_fc = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps),len(lead_times)])
spr_an = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps)])
err_fc = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps),len(lead_times)])
err_an = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps)])
rmse_fc = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps),len(lead_times)])
rmse_an = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps)])
crps_fc = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps),len(lead_times)])
crps_an = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps)])
sprrmse_ratio_an = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps)])
sprrmse_ratio_fc = np.empty([Neq+1,len(loc),len(add_inf),len(rtpp),len(rtps),len(lead_times)])
OI = np.empty([Neq+2,len(loc),len(add_inf),len(rtpp),len(rtps)])

indices = list(itertools.product(list(range(0,len(loc))), list(range(0,len(add_inf))), list(range(0,len(rtpp))), list(range(0,len(rtps)))))

for m in range(0,len(indices)):
    try:
        i = indices[m][0]
        j = indices[m][1]
        k = indices[m][2]
        l = indices[m][3]
        print((i,j,k,l))
        spr_an[:,i,j,k,l], err_an[:,i,j,k,l], rmse_an[:,i,j,k,l], crps_an[:,i,j,k,l], sprrmse_ratio_an[:,i,j,k,l], OI[:,i,j,k,l] = ave_stats_an(Nk_fc,Neq,n_d,n_ens,Nmeas,spin_up,indices[m],outdir,loc,add_inf,rtpp,rtps,X_tr)
        for n in range(0,len(lead_times)):
            print(i,j,k,l,n)
            spr_fc[:,i,j,k,l,n], err_fc[:,i,j,k,l,n], rmse_fc[:,i,j,k,l,n], crps_fc[:,i,j,k,l,n], sprrmse_ratio_fc[:,i,j,k,l,n] = ave_stats_fc(Nk_fc,Neq,n_d,n_ens,Nmeas,spin_up,indices[m],outdir,loc,add_inf,rtpp,rtps,lead_times[n],X_tr)
    except IOError as err:
        continue

##################################################################
print(' *** PLOT: STATS matrix with spread and RMSE ***')
##################################################################

# Compute min/max for forecast spread and RMSE.
for n in range(len(lead_times)):
#x = np.ma.masked_invalid([err_fc, rmse_fc, spr_fc, err_an, rmse_an, spr_an])
    for m in range(Neq+1):
        x = np.ma.masked_invalid([rmse_fc[m,:,:,:,:,n]])
        y = np.ma.masked_invalid([spr_fc[m,:,:,:,:,n]])
    
        # Choose the colour map limits.
        minvalx = np.around(np.amin(x),
                       decimals=3)
        maxvalx = np.around(np.amax(x),
                       decimals=3)
        # Choose the colour map limits.
        minvaly = np.around(np.amin(y),
                       decimals=3)
        maxvaly = np.around(np.amax(y),
                       decimals=3)
    
        # Rearrange data in the order loc, rtps, add_inf and mask out nan values.
        summary_plot(np.ma.masked_invalid(np.swapaxes(spr_fc[m,:,:,0,:,n], 0, 2)), 'spr',
                 'Forecast spread ('+var[m]+') +'+str(lead_times[n])+'h (SPR)', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minvaly, maxvaly, str(figsdir + '/spr_'+var[m]+'_fc_+'+str(lead_times[n])+'.pdf'),cm.YlOrRd)

        summary_plot(np.ma.masked_invalid(np.swapaxes(rmse_fc[m,:,:,0,:,n], 0, 2)), 'rmse', 
                 'Forecast error ('+var[m]+') +'+str(lead_times[n])+'h (RMSE)', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minvalx, maxvalx, str(figsdir + '/rmse_'+var[m]+'_fc_+'+str(lead_times[n])+'.pdf'),cm.YlOrRd)


# Compute min/max for analysis spread and RMSE.
for m in range(Neq+1):
    x = np.ma.masked_invalid([rmse_an[m,:,:,:,:], spr_an[m,:,:,:,:]])

    # Choose the colour map limits.
    minval = np.around(np.amin(x),
                       decimals=3)
    maxval = np.around(np.amax(x),
                       decimals=3)

    summary_plot(np.ma.masked_invalid(np.swapaxes(rmse_an[m,:,:,0,:], 0, 2)),'rmse', 
                 'Analysis error ('+var[m]+', RMSE)', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minval, maxval, str(figsdir + '/rmse_'+var[m]+'_an.pdf'),cm.YlOrRd)

    summary_plot(np.ma.masked_invalid(np.swapaxes(spr_an[m,:,:,0,:], 0, 2)), 'spr',
                 'Analysis spread (SPR)', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minval, maxval, str(figsdir + '/spr_'+var[m]+'_an.pdf'),cm.YlOrRd)

##################################################################
print(' *** PLOT: STATS matrix with MAE ***')
##################################################################

for m in range(Neq+1):
# Compute min/max for MAE.
    for n in range(len(lead_times)):
        x = np.ma.masked_invalid([err_fc[m,:,:,:,:,n]])

        # Choose the colour map limits.
        minval = np.around(np.amin(x),
                       decimals=4)
        maxval = np.around(np.amax(x),
                       decimals=4)

        summary_plot(np.ma.masked_invalid(np.swapaxes(err_fc[m,:,:,0,:,n], 0, 2)), 'mae',
                 'Forecast error ('+var[m]+') +'+str(lead_times[n])+'h (MAE)', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minval, maxval, str(figsdir + '/err_'+var[m]+'_fc_+'+str(lead_times[n])+'.pdf'),cm.YlOrRd)

    y = np.ma.masked_invalid([err_an[m,:,:,:,:]])

    # Choose the colour map limits.
    minval = np.around(np.amin(y),
                       decimals=4)
    maxval = np.around(np.amax(y),
                       decimals=4)

    summary_plot(np.ma.masked_invalid(np.swapaxes(err_an[m,:,:,0,:], 0, 2)), 'mae',
             'Analysis error ('+var[m]+', MAE)', ['RTPS', '$\gamma_a$', '$L_{loc}$'],

             [rtps, add_inf, loc], minval, maxval, str(figsdir + '/err_'+var[m]+'_an.pdf'),cm.YlOrRd)


##################################################################
print(' *** PLOT: STATS matrix with CRPS ***')
##################################################################

for m in range(Neq+1):
    # Compute min/max for CRPS.
    for n in range(len(lead_times)):
        x = np.ma.masked_invalid([crps_fc[m,:,:,:,:,n]])

        # Choose the colour map limits.
        minval = np.around(np.amin(x),
                       decimals=4)
        maxval = np.around(np.amax(x),
                       decimals=4)

        summary_plot(np.ma.masked_invalid(np.swapaxes(crps_fc[m,:,:,0,:,n], 0, 2)), 'crps', 
             'Forecast ('+var[m]+') +'+str(lead_times[n])+'h CRPS', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
             [rtps, add_inf, loc], minval, maxval, str(figsdir + '/crps_'+var[m]+'_fc_+'+str(lead_times[n])+'.pdf'),cm.YlOrRd)

    y = np.ma.masked_invalid([crps_an[m,:,:,:,:]])

    minval = np.around(np.amin(y),
                   decimals=4)
    maxval = np.around(np.amax(y),
                   decimals=4)

    summary_plot(np.ma.masked_invalid(np.swapaxes(crps_an[m,:,:,0,:], 0, 2)), 'crps',
                 'Analysis ('+var[m]+') CRPS', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minval, maxval, str(figsdir + '/crps_'+var[m]+'_an.pdf'),cm.YlOrRd)

##################################################################
print(' *** PLOT: OI matrix ***')
##################################################################

for m in range(Neq+2):
    # Compute min/max for CRPS.
    x = np.ma.masked_invalid(OI[m,:,:,:,:])

    # Choose the colour map limits.

    minval = 10.
    maxval = 50.

    summary_plot(np.ma.masked_invalid(np.swapaxes(OI[m,:,:,0,:], 0, 2)), 'OI',
                 'Observational Impact ('+var_oi[m]+')', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minval, maxval, str(figsdir + '/oi_'+var_oi[m]+'.pdf'),cm.BrBG)

##################################################################
print(' *** PLOT: SPREAD-RMSE DIFFERENCE ***')
##################################################################

# Compute min/max for SPREAD-RMSE.
#for n in range(len(lead_times)):
    #x = np.ma.masked_invalid([sprrmse_diff_fc[:,:,:,:,n], sprrmse_diff_an])

# Choose the colour map limits.
#minval = np.around(np.amin(x),
#                   decimals=2)
#maxval = np.around(np.amax(x),
#                   decimals=2)
#    minval = -0.08
#    maxval = 0.08

#    summary_plot(np.ma.masked_invalid(np.swapaxes(sprrmse_diff_fc[:,:,0,:,n], 0, 2)),
#             'Forecast +'+str(lead_times[n])+'h SPR-RMSE', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
#             [rtps, add_inf, loc], minval, maxval, str(figsdir + '/sprrmse_diff_fc_+'+str(lead_times[n])+'.pdf'),cm.BrBG)

#summary_plot(np.ma.masked_invalid(np.swapaxes(sprrmse_diff_an[:,:,0,:], 0, 2)),
#             'Analysis SPR-RMSE', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
#             [rtps, add_inf, loc], minval, maxval, str(figsdir + '/sprrmse_diff_an.pdf'),cm.BrBG)

##################################################################
print(' *** PLOT: SPREAD TO RMSE RATIO ***')
##################################################################

for m in range(Neq+1):
    
    minval = 0.5
    maxval = 1.5

    # Compute min/max for SPREAD-RMSE.
    for n in range(len(lead_times)):
    
        summary_plot(np.ma.masked_invalid(np.swapaxes(sprrmse_ratio_fc[m,:,:,0,:,n], 0, 2)),'sprrmse_ratio',
                 'Forecast ('+var[m]+') +'+str(lead_times[n])+'h SPR/RMSE', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minval, maxval, str(figsdir + '/sprrmse_ratio_'+var[m]+'_fc_+'+str(lead_times[n])+'.pdf'),cm.BrBG)

    summary_plot(np.ma.masked_invalid(np.swapaxes(sprrmse_ratio_an[m,:,:,0,:], 0, 2)),'sprrmse_ratio',
                 'Analysis ('+var[m]+') SPR/RMSE', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
                 [rtps, add_inf, loc], minval, maxval, str(figsdir + '/sprrmse_ratio_'+var[m]+'_an.pdf'),cm.BrBG)

##################################################################
#print(' *** PLOT: FORECAST-ANALYSY RMSE***')
##################################################################
# Compute min/max for SPREAD-RMSE.
#x = np.ma.masked_invalid(fcan_diff_rmse)

# Choose the colour map limits.
#minval = np.around(np.amin(x),
#                   decimals=int(abs(np.floor(np.log10(np.amin(x))))))
#maxval = np.around(np.amax(x),
  #                 decimals=int(abs(np.floor(np.log10(np.amax(x))))))
#minval = 0.

#summary_plot(np.ma.masked_invalid(np.swapaxes(fcan_diff_rmse[:,:,0,:], 0, 2)),
 #            'Forecast-analysis RMSE', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
 #            [rtps, add_inf, loc], minval, maxval, str(figsdir + '/fcan_diff_rmse.pdf'),cm.bone)

##################################################################
#print(' *** PLOT: FORECAST-ANALYSY CRPS***')
##################################################################
# Compute min/max for SPREAD-RMSE.
#x = np.ma.masked_invalid(fcan_diff_crps)

# Choose the colour map limits.
#minval = np.around(np.amin(x),
#                   decimals=int(abs(np.floor(np.log10(np.amin(x))))))
#maxval = np.around(np.amax(x),
 #                  decimals=int(abs(np.floor(np.log10(np.amax(x))))))

#minval = 0.

#summary_plot(np.ma.masked_invalid(np.swapaxes(fcan_diff_crps[:,:,0,:], 0, 2)),
#             'Forecast-analysis CRPS', ['RTPS', '$\gamma_a$', '$L_{loc}$'],
#             [rtps, add_inf, loc], minval, maxval, str(figsdir + '/fcan_diff_crps.pdf'),cm.bone)
