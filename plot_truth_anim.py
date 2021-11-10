import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import importlib.util
import sys
from isen_func import *
import animatplot as amp

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

U_tr = np.load(outdir+str('/U_hovplt.npy'))
timevec = np.load(outdir+str('/t_hovplt.npy'))

### OLD ANIMATION

# Set up formatting for the movie files
#Writer = animation.writers['imagemagick']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#def set_frame(i):

#    return U_tr[0,:,i]

#U_tr = np.load(outdir+str('/U_array_Nk'+str(Nk)+'.npy'))

#fig, axes = plt.subplots(4, 1, figsize=(8,8))
#axes[0].set_ylim(0.,0.4)
#axes[0].hlines(sig_c,-10.,Nk+10.,colors='gray',linestyles='dashed')
#axes[0].hlines(sig_r,-10.,Nk+10.,colors='gray',linestyles='dashed')

#ims = []

#for i in range(np.size(U_tr,2)):
#    im0, = axes[0].plot(U_tr[0,:,i],'b')
#    im1, = axes[1].plot(U_tr[1,:,i]/U_tr[0,:,i],'b')
#    im2, = axes[2].plot(U_tr[2,:,i]/U_tr[0,:,i],'b')
#    im3, = axes[3].plot(U_tr[3,:,i]/U_tr[0,:,i],'b')
#    ims.append([im0,im1,im2,im3])

#anim = animation.ArtistAnimation(fig,ims,interval=500,repeat=False)
#anim.save(outdir+'/timeseries.gif',writer=writer)

###

### NEW ANIMATION

t = timevec[:22498:28]

fig, axes = plt.subplots(4, 1, figsize=(8,8))
axes[0].axhline(y=0.21,xmin=0,xmax=1,linestyle='--',color='blue')
axes[0].axhline(y=0.24,xmin=0,xmax=1,linestyle='--',color='red')
axes[0].set_ylim([0.1,0.35])
axes[0].set_ylabel('$\sigma$(x)',fontsize=15)
axes[0].tick_params(labelsize=15)
axes[0].set_xticks([])
axes[1].set_ylim([-0.2,0.2])
axes[1].set_ylabel('u(x)',fontsize=15)
axes[1].set_xticks([])
axes[1].tick_params(labelsize=15)
axes[2].set_ylim([-0.5,0.5])
axes[2].set_ylabel('v(x)',fontsize=15)
axes[2].set_xticks([])
axes[2].tick_params(labelsize=15)
axes[3].set_ylim([0.,0.1])
axes[3].set_ylabel('r(x)',fontsize=15)
axes[3].set_xticks([0,199,399])
axes[3].set_xticklabels(['0','0.5','1'])
axes[3].set_xlabel('x',fontsize=15)
axes[3].get_yaxis().set_label_coords(-0.1,0.5)
axes[3].tick_params(labelsize=15)

sigline_block = amp.blocks.Line(range(400),U_tr[0,:,:22498:28],ax=axes[0], t_axis=1, linewidth = 1.0, drawstyle='steps-post')
uline_block = amp.blocks.Line(range(400),U_tr[1,:,:22498:28],ax=axes[1], t_axis=1, linewidth = 1.0, drawstyle='steps-post')
vline_block = amp.blocks.Line(range(400),U_tr[2,:,:22498:28],ax=axes[2], t_axis=1, linewidth = 1.0, drawstyle='steps-post')
rline_block = amp.blocks.Line(range(400),U_tr[3,:,:22498:28],ax=axes[3], t_axis=1, linewidth = 1.0, drawstyle='steps-post')

timeline = amp.Timeline(t, fps=1000)

# now to construct the animation
anim = amp.Animation([sigline_block, uline_block, vline_block, rline_block], timeline)
anim.controls()

anim.save_gif(outdir+str('/model_anim'))

plt.show()

