import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys

##################################################################
##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

outdir = config.outdir

Qmatrix = np.load(outdir+'/Qmatrix.npy')
Qdiag = np.diagonal(Qmatrix)

plt.plot(Qdiag)
plt.xlabel('vector index',fontsize=15)
plt.ylabel('var($\eta$)',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.vlines(199,0,0.00008,color='black',linestyle='dashed')
plt.vlines(399,0,0.00008,color='black',linestyle='dashed')
plt.vlines(599,0,0.00008,color='black',linestyle='dashed')
plt.show()
