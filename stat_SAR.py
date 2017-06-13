# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:30:09 2015

@author: capdessus
stat_SAR: génération de donnnées synthétques RADAR"""

import numpy as np
import numpy.random as rnd
import scipy.linalg as lg
import basic_lib as bl
import load_param as lp

def generate_UPS(param):
    """genere la matrice de covariance Ups dans le cadre du modèle RVoG en SB
    pour un jeu de paramètre donné   
     """
    ind_baseline = 0
    k_z = param.k_z[ind_baseline]
    theta = param.theta
    T_vol = param.T_vol
    T_ground = param.T_ground
    h_v = param.h_v
    z_g = param.z_g
    sigma_v = param.sigma_v
    
    alpha=2*sigma_v/np.cos(theta)
    a=np.exp(-alpha)    
    I1=(1-a)/alpha
    I2=(np.exp(1j*k_z*h_v)-a)/ \
        (1j*k_z+alpha)
    #Construction matrice de covariance Upsilon
    Ups=bl.nans([6,6],dtype=np.complex)
    T=bl.nans([3,3],dtype=np.complex)
    Om=bl.nans([3,3],dtype=np.complex)
    
    T=I1*T_vol+a*T_ground
    Om=np.exp(1j*k_z*z_g)*(I2*T_vol+a*T_ground)
    Ups = np.vstack([np.hstack([T,Om]),np.hstack([Om.T.conj(),T])])    
    return Ups
    
def generate_PolInSAR(Ups,N):
    """génère N vecteurs d'observation PolInSAR (donc de taille ncxN, nc
        nombre de composantes (3x2 en SB) du vec d'observation)
       chaque vecteur eest une realisation d'une loi gaussienne circulaire 
       centrée de matrice de covariance Ups
    """   
  
    N = int(N)
    nc = int(Ups.shape[0])
    sqrt_Ups = lg.sqrtm(Ups)
    mean = [0 for i in range(nc)]
    
    cov = np.eye(nc)
    
    r_mat_rand = rnd.multivariate_normal(mean,cov,N) #1 real par ligne du vec k (Nxnc) 
    i_mat_rand = rnd.multivariate_normal(mean,cov,N) #1 real par ligne du vec k (Nxnc) 
    vec = np.sqrt(0.5)*sqrt_Ups.dot(r_mat_rand.T + 1j*i_mat_rand.T)
    #Upsilon_moy = vec.dot(vec.T.conj()) / (N-1) 
    return vec


def launch_test():
    
    Na = 3
    Np = 3
    A = 0.75
    E = 71
    k_z = [0.1,0.2]
    theta=45*np.pi/180
    hv=30
    sigmav=np.log(10**(0.4/20))
    zg=1
    #Test avec matrice sans bruit de speckle
    #SB
    param_MB = lp.load_param('DB_1')
    Ups=param_MB.get_upsilon()
    N=1000
    vec=generate_PolInSAR(Ups,N)    
    #plt.plot(np.real(vec),np.imag(vec),'.b')

if __name__ == '__main__':
    print ''
    

    
    
    
    