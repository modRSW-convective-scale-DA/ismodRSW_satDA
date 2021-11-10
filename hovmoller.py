import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import importlib.util
import sys

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir

### IMPORT DATA ###
U = np.load(outdir+'/U_hovplt.npy') 
x = np.arange(0.,1.,0.0025)
t = np.load(outdir+'/t_hovplt.npy')

print(t.shape,x.shape)
print(t[46940])

# Start figure
fig = plt.figure(figsize=(10,10)) 

# Top plot for density
ax1 = fig.add_subplot(221)
#siglevs = np.array([0.,0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.3, 0.4, 0.5])
siglevs = np.array([0.,0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3])
#clcol = plt.cm.Greys(np.linspace(0.,0.5,20))
#racol = plt.cm.YlOrBr(np.linspace(0.,1.,68))
#white = np.tile([1.,1.,1.,1.],(25,1))
clcol = plt.cm.Greys(np.linspace(0.,0.5,40))
racol = plt.cm.YlOrBr(np.linspace(0.,1.,15))
white = np.tile([1.,1.,1.,1.],(20,1))
cols = np.vstack((white,clcol,racol))
cm = mpl.colors.LinearSegmentedColormap.from_list('pippo', cols)
cv1 = ax1.contourf(x,t[:46940:20],U[0,:,:46940:20].T,siglevs,cmap=cm)
cs1 = ax1.contour(x,t[:46940:20],U[0,:,:46940:20].T,siglevs,colors='k',linewidths=0.1)
cbar = plt.colorbar(cv1,ax=ax1,ticks=siglevs,orientation='vertical')
cbar.ax.set_title('$\sigma$',fontsize=20)
cbar.ax.tick_params(labelsize=14)
ax1.set_ylabel('t',fontsize=20)
ax1.tick_params(labelsize=14)
ax1.set_xticklabels([])

# Middle plot for hor. velocity
ax2 = fig.add_subplot(223)
ulevs = np.arange(-1.0,1.2,0.2)
cv2 = ax2.contourf(x,t[:46940:20],(U[1,:,:46940:20]/U[0,:,:46940:20]).T,ulevs,cmap=plt.cm.bwr)
cs2 = ax2.contour(x,t[:46940:20],(U[1,:,:46940:20]/U[0,:,:46940:20]).T,ulevs,colors='k',linewidths=0.5)
cbar = plt.colorbar(cv2,ax=ax2,ticks=ulevs,orientation='vertical')
cbar.ax.set_title('u',fontsize=20)
cbar.ax.tick_params(labelsize=14)
ax2.set_ylabel('t',fontsize=20)
ax2.set_xlabel('x',fontsize=20)
ax2.tick_params(labelsize=14)
ax2.set_xticks([0.0,0.25,0.5,0.75,1.0])
ax2.set_xticklabels([0.0,0.25,0.5,0.75,1.0])

# Middle plot for mer. velocity
ax3 = fig.add_subplot(224)
vlevs = np.arange(-1.0,1.2,0.2)
cv3 = ax3.contourf(x,t[:46940:20],(U[2,:,:46940:20]/U[0,:,:46940:20]).T,vlevs,cmap=plt.cm.bwr)
cs1 = ax3.contour(x,t[:46940:20],(U[2,:,:46940:20]/U[0,:,:46940:20]).T,vlevs,colors='k',linewidths=0.5)
cbar = plt.colorbar(cv3,ax=ax3,ticks=vlevs,orientation='vertical')
cbar.ax.set_title('v',fontsize=20)
cbar.ax.tick_params(labelsize=14)
ax3.set_xlabel('x',fontsize=20)
ax3.tick_params(labelsize=14)
ax3.set_xticks([0.0,0.25,0.5,0.75,1.0])
ax3.set_xticklabels([0.0,0.25,0.5,0.75,1.0])
ax3.set_yticklabels([])

# Bottom plot for rain 
ax4 = fig.add_subplot(222)
rlevs = np.array([0.0,0.01,0.02,0.03,0.04,0.05,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.25,0.3,0.35])
cm = plt.cm.get_cmap("YlGnBu")
cm.set_bad("white")
cv4 = ax4.contourf(x,t[:46940:20],np.ma.masked_where((U[3,:,:46940:20]/U[0,:,:46940:20])<5e-3,(U[3,:,:46940:20]/U[0,:,:46940:20])).T,rlevs,cmap=cm)
cs4 = ax4.contour(x,t[:46940:20],(U[3,:,:46940:20]/U[0,:,:46940:20]).T,rlevs,colors='k',linewidths=0.1)
cbar = plt.colorbar(cv4,ax=ax4,ticks=rlevs,orientation='vertical')
cbar.ax.set_title('r',fontsize=20)
cbar.ax.tick_params(labelsize=14)
ax4.tick_params(labelsize=14)
ax4.set_xticklabels([])
ax4.set_yticklabels([])

plt.savefig(outdir+'/hovplt_truth.png')
plt.show()
