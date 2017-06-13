# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 11:57:00 2016

@author: pcapdess
Module contenant les fonction de contraintes pour l'optimisation de 
mlogV_zg_known()
"""

from __future__ import division

import numpy as np
import numpy.linalg as npl  
import RVoG_MB as mb

def constr_Tvol_posit(X):
    """Fonction contrainte : est positive lorsque la matrice Tvol est semi
    def positive ie le min des vap est positif
    X=(Tvol,Tgro,hv,sigv,{gtij})"""
    vec_Tvol = X[0:9]
    Tvol = mb.get_Tm_from_vecTm(vec_Tvol)    
    eigTvol,_ = npl.eig(Tvol)
    return np.min(np.real(eigTvol))

def constr_Tground_posit(X):
    """Fonction contrainte : est positive lorsque la matrice Tground est semi
    def positive ie le min des vap est positif
    X=(Tvol,Tgro,hv,sigv,{gtij})"""
    vec_Tground = X[9:18]
    Tground = mb.get_Tm_from_vecTm(vec_Tground)    
    eigTground,_ = npl.eig(Tground)
    return np.min(np.real(eigTground))
    
def constr_hv_posit(X):
    """hv>0"""
    return X[18]
    
def constr_sigv_posit(X):
    """sigv>=0"""
    return X[19]

def constr_gt12_inf1(X):
    """gt12<=1"""
    gt12 = X[20]
    return 1-gt12
    
def constr_gt12_posit(X):
    """0<=gt12"""
    gt12 = X[20]
    return gt12
    
def constr_gt13_inf1(X):
    """gt13<=1"""
    gt13 = X[21]
    return 1-gt13

def constr_gt13_posit(X):
    """0<=gt13"""
    gt13 = X[21]
    return gt13

def constr_gt23_inf1(X):
    """gt23<=1"""
    gt23 = X[22]
    return 1-gt23
    
def constr_gt23_posit(X):
    """0<gt23"""
    gt23 = X[22]
    return gt23
    


    