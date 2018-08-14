import numpy as np
import matplotlib.pyplot as plt


Ne=16
data_folder = 'data/'
filenameIpt = data_folder + 'x_normalNO_'  + str(Ne) + '.dat'
filenameOpt = data_folder + 'psi_normalNO_' + str(Ne) + '.dat'

xraw  = np.loadtxt(filenameIpt, delimiter=None, usecols=range(Ne))
yraw  = np.loadtxt(filenameOpt, delimiter=None, usecols=range(1))

#choose a random sample
sample = 100
xtail = xraw[sample,2:]

def phi_normal(x,Nphi):
  #this function is for scalar x
  center = 0
  sigma = 1
  centers = center + np.arange(Nphi)*0.5*sigma
  phi = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-centers)**2/(2*sigma**2))
  return phi
  
def phi_func(x):
  return phi_normal(x,Ne)
  
def slater_det_evaluate(phi_func,x):
  Nphi = Ne
  phi_val_mat = np.zeros((Nphi,Nphi))
  # Evaluate the value of the orbitals  phi_j(x_i)
  for i in range(0, Nphi-1):
      phi_val_mat[:,i] = phi_func(x[i])
  return np.linalg.det(phi_val_mat)
  
def H_evaluate(phi_func,x):
  slater = slater_det_evaluate(phi_func,x)
  return -np.log((slater)**2)


  