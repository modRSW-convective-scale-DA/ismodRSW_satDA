from builtins import str
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from parameters import *
from isen_func import *


U_tr = np.load(str('/output/test_model/U_array_Nk800.npy'))

fig, axes = plt.subplots(4, 1, figsize=(8,8))

axes[0].set_ylim(0.,0.5)
axes[0].hlines(sig_c,-10.,810.,colors='gray',linestyles='dashed')
axes[0].hlines(sig_r,-10.,810.,colors='gray',linestyles='dashed')
axes[0].set_ylabel('$\sigma$(x)')
axes[0].plot(U_tr[0,:,0],'b')
axes[1].set_ylabel('u(x)')
axes[1].plot(U_tr[1,:,0],'b')
axes[2].set_ylabel('v(x)')
axes[2].plot(U_tr[2,:,0],'b')
axes[3].set_ylabel('r(x)')
axes[3].plot(U_tr[3,:,0],'b')
#plt.show()
plt.savefig('/nobackup/mmlca/figs/ic_trans_jet+zon_vel.png')
