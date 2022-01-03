### This script generates a look-up table to convert pseudo-density into non-dimensional pressure.

##################################################################
# GENERIC MODULES REQUIRED
##################################################################
import numpy as np
import h5py
import importlib
import sys
sys.path.append('..')

##################################################################
# IMPORT PARAMETERS FROM CONFIGURATION FILE
##################################################################
spec = importlib.util.spec_from_file_location("config", sys.argv[1])
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

R = config.R
cp = config.cp
theta2 = config.theta2
theta1 = config.theta1
eta0 = config.eta0
Z0 = config.Z0
g = config.g
k = config.k

# Define additional parameter
kinv = 1./k

# Parameters for inversion
t_res=0.001 
res=10000000

# Values of pseudo-density to be converted
sigma = np.arange(0.,2.,1e-4)

# Residual between computed and tabulated values of sigma given non-dimensional pressure eta 
sig_zero = (lambda etab: etab-((1./(R*kinv))*(1./(theta1-theta2))*(-(R*kinv)*theta2*etab**k+(R*kinv)*theta1*eta0**k+g*Z0))**kinv-sigma)

# Interval of values of non-dimensional pressure within which inversion is performed
etab_zero = np.linspace(1.0,1.5,res)

# Compute residual for the first element of etab_zero
sig = sig_zero(etab_zero[0])

# Array of inverted values of non-dimensional pressure (one for each sigma in input)
etab = np.zeros(len(sigma))
i = 1

# Loop over the array of sigma to invert
for j in range(0,len(sigma)):
    # Loop over values of the residuals until a value smaller than res is found
    while abs(sig[j])>t_res:
        sig[j] = sig_zero(etab_zero[i])[j]
        i = i + 1        
        # Check if the loop has reached the end of the array
        if(i>=res):
            print('Inversion has failed. Try a different interval (or resolution) for etab_zero or a different t_res')
            exit(1)
    # Save and print inverted value of non-dimensional pressure
    etab[j] = etab_zero[i]
    print(etab[j])

### Create table for storage
array2store = np.array([sigma,etab])

### Define metadata for output hdf file
file_name = 'sigma_eta_theta2_'+str(int(theta2))+'_theta1_'+str(int(theta1))+'_eta0_'+str(eta0)+'_Z0_'+str(int(Z0))+'_k_'+str(round(k,2))+'.hdf'
with h5py.File(file_name, 'w') as outfile:
    dataset = outfile.create_dataset('sigma_eta_iversion_table',data=array2store)
    dataset.attrs['theta2'] = theta2
    dataset.attrs['theta1'] = theta1
    dataset.attrs['eta0'] = eta0
    dataset.attrs['Z0'] = Z0
    dataset.attrs['R/cp'] = k

### Print final message
print('Look-up table written in file: ', file_name)
