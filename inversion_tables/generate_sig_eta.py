import numpy as np
import h5py

R = 287.
cp = 287.
theta2 = 291.8
theta1 = 311.
eta0 = 0.48
Z0 = 6120.
g = 9.81
k = R/cp
kinv = 1./k

sigma = np.arange(0.,2.,1e-4)
sig_zero = (lambda etab: etab-((1./(R*kinv))*(1./(theta1-theta2))*(-(R*kinv)*theta2*etab**k+(R*kinv)*theta1*eta0**k+g*Z0))**kinv-sigma)
etab_zero = np.linspace(1.1,1.5,10000000)
sig = sig_zero(etab_zero[0])
etab = np.zeros(len(sigma))
i = 1
for j in range(0,len(sigma)):
    while abs(sig[j])>0.001:
        sig[j] = sig_zero(etab_zero[i])[j]
        i = i + 1
    etab[j] = etab_zero[i]
    print(etab[j])

array2store = np.array([sigma,etab])

with h5py.File('/home/home02/mmlca/r1012_sat_modrsw_enkf/isenRSW_EnKF_py3/inversion_tables/sigma_eta_theta2_'+str(int(theta2))+'_theta1_'+str(int(theta1))+'_k=1_'+'.hdf', 'w') as outfile:
    dataset = outfile.create_dataset('sigma_eta_iversion_table',data=array2store)
    dataset.attrs['theta2'] = theta2
    dataset.attrs['theta1'] = theta1
    dataset.attrs['eta0'] = eta0
    dataset.attrs['Z0'] = Z0
    dataset.attrs['k'] = k
