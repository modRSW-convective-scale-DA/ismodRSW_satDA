import sys
import numpy as np
import matplotlib.pyplot as plt

file_1 = sys.argv[1]
file_2 = sys.argv[2]

U_tr1 = np.load(file_1)
U_tr2 = np.load(file_2)

fig, axes = plt.subplots(4, 1, figsize=(8,8))
axes[0].plot(range(400),U_tr1[0,:,1])
axes[0].plot(range(0,400,2),U_tr2[0,:,1])
axes[0].set_ylabel('$\sigma$(x)',fontsize=15)
axes[0].set_xticks([])
axes[0].get_yaxis().set_label_coords(-0.1,0.5)
axes[0].tick_params(labelsize=15)
axes[1].plot(range(400),U_tr1[1,:,1])
axes[1].plot(range(0,400,2),U_tr2[1,:,1])
axes[1].set_ylabel('u(x)',fontsize=15)
axes[1].get_yaxis().set_label_coords(-0.1,0.5)
axes[1].set_xticks([])
axes[1].tick_params(labelsize=15)
axes[2].plot(range(400),U_tr1[2,:,1])
axes[2].plot(range(0,400,2),U_tr2[2,:,1])
axes[2].set_ylabel('v(x)',fontsize=15)
axes[2].set_xticks([])
axes[2].get_yaxis().set_label_coords(-0.1,0.5)
axes[2].tick_params(labelsize=15)
axes[3].plot(range(400),U_tr1[3,:,1])
axes[3].plot(range(0,400,2),U_tr2[3,:,1])
axes[3].set_ylabel('r(x)',fontsize=15)
axes[3].set_xticks([0,199,399])
axes[3].set_xticklabels(['0','0.5','1'])
axes[3].set_xlabel('x',fontsize=15)
axes[3].get_yaxis().set_label_coords(-0.1,0.5)
axes[3].tick_params(labelsize=15)
plt.show()
