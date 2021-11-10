'''
    A few checks on characteristics of the nature run: plot all trajectories, check height and  rain extremes
    '''
from __future__ import print_function
from builtins import range
import numpy as np
import os
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.cm as cm
from parameters import *

U_tr = np.load('/home/home02/mmlca/test_model/U_array_Nk800.npy')
print('Shape of truth array is:', np.shape(U_tr))
print('Maximum height reached is:', np.max(U_tr[0,:,:]))
print('Minimum height reached is:', np.min(U_tr[0,:,:]))

print('Max rain is:', np.max(U_tr[3,:,:]/U_tr[0,:,:]))
print('Min rain is:', np.min(U_tr[3,:,:]/U_tr[0,:,:]))

colors = cm.rainbow(np.linspace(0, 1, len(U_tr[0,0,:])))

figure()
for i in range(1,len(U_tr[0,0,:]+1)):
	plt.plot(U_tr[0,:,i],label='+%i'%(i),color=colors[i])
#plot(U_tr[0,:,-1],linewidth = 3)
#plt.colorbar()
#plt.legend(loc='top center',ncol=16)
show()
