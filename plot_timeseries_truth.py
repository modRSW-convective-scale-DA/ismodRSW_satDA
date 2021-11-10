import numpy as np
import sys
import importlib.util
import matplotlib.pyplot as plt

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

Nk = config.Nk
outdir = config.outdir
sig_c = config.sig_c
sig_r = config.sig_r

### LOAD DATA
U_tr = np.load(outdir+str('/U_hovplt.npy'))
timevec = np.load(outdir+str('/t_hovplt.npy'))

t = timevec

fig, axes = plt.subplots(4, 1, figsize=(8.5,9))
axes[0].plot(t[:46940:20],U_tr[0,199,:46940:20],color='gray')
axes[0].hlines(sig_c,t[0],t[46940],color='green',linestyle='dashed',label='$\sigma_c$, convection threshold')
axes[0].hlines(sig_r,t[0],t[46940],color='red',linestyle='dashed',label='$\sigma_r$, rain threshold')
axes[0].set_ylabel('$\sigma(x_0)$',fontsize=18)
axes[0].tick_params(axis='y',labelsize=15)
axes[0].set_xticks([])
axes[0].yaxis.set_label_coords(-0.11,0.5)
axes[1].plot(t[:46940:20],U_tr[1,199,:46940:20],color='gray')
axes[1].set_ylabel('$u(x_0)$',fontsize=18)
axes[1].tick_params(axis='y',labelsize=15)
axes[1].set_xticks([])
axes[1].yaxis.set_label_coords(-0.11,0.5)
axes[2].plot(t[:46940:20],U_tr[2,199,:46940:20],color='gray')
axes[2].set_ylabel('$v(x_0)$',fontsize=18)
axes[2].tick_params(axis='y',labelsize=15)
axes[2].set_xticks([])
axes[2].yaxis.set_label_coords(-0.11,0.5)
axes[3].plot(t[:46940:20],U_tr[3,199,:46940:20],color='gray')
axes[3].set_ylabel('$r(x_0)$',fontsize=18)
axes[3].tick_params(axis='both',labelsize=15)
axes[3].set_xlabel('t',fontsize=18)
axes[3].yaxis.set_label_coords(-0.11,0.5)
handles, labels = axes[3].get_legend_handles_labels()
fig.legend(loc='upper center',ncol=2,fontsize=12)
plt.show()
