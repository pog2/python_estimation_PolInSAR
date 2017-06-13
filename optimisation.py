# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:16:46 2015

@author: capdessus
"""
from __future__ import division
import plot_tomo as pt
from collections import deque
import numpy as np
import scipy.optimize as opti
import RVoG_MB as mb
import matplotlib.pyplot as plt
import numpy.linalg as npl
import contraintes as c 
import load_param as lp
import estimation as e 
import basic_lib as bl

def sum_diff(vec_X):
    """Retourne la somme de toutes les differences (\| \|²) possibles du vecteur vec_X"""
    
    size= np.size(vec_X)
    mat_permX= np.zeros((size,size),dtype=complex)
    mat_diffX = np.zeros((size-1,size),dtype=complex)
    vec_perm = deque(vec_X)
    for i in range(size):
        mat_permX[i,:]=np.array(vec_perm)        
        vec_perm.rotate()
    for i in range(size-1):
        mat_diffX[i,:]=(np.abs(mat_permX[0,:]-mat_permX[i+1,:]))**2
    S = np.sum(np.sum(mat_diffX))
    return S

def min_sum_diff(mat_X):
    """Calcule sum_diff(mat_X(choix(:,i))) pour tous les choix possibles
    sotckés dans mat_choix"""
    
    sze = mat_X.shape # Na x nbre_gammat
    Na = sze[0]
    nb_gt = sze[1]
    mat_choix = gene_mat_ind_choice(Na,nb_gt)
    S = np.zeros(nb_gt**Na)#nbre de choix possibles=nbre_gammat**Na
    for i in range(nb_gt**Na):
        vec_chx = mat_choix[:,i]
        vec_X = np.array([mat_X[j,vec_chx[j]] for j in range(Na)])
        S[i] = sum_diff(vec_X)
    minS=np.min(S)
    idx_min=np.argmin(S)
    chx_min=mat_choix[:,idx_min]
    return S,idx_min,minS,chx_min
    

    
def gene_mat_ind_choice(Na,m):
    """Genère une matrice avec tous les vecteurs d'indices possibles 
    pour un vecteur de taille Na dont chaque composante peut prendre
    m valeurs (0 .. m-1) soit m choix possibles.
    
    **Entrées**
        * *Na* : nombre d'antennes (acquisitions)
        * *m* : nombre de choix possibles pour une composante
        
    **Sortie**
        * *matM* : matrice Nax(m^Na)"""
        
    matM = np.zeros((Na,m**Na))
    for i in range(m**Na):
        matM[:,i]=np.array([np.floor(i/m**j)%m for j in range(Na)])
    return matM

    
 
if __name__=='__main__':
    
    
    param = lp.load_param('DB_3')
    param.h_v=30
    param.gammat = np.ones((3,3))*0.95
    
    verbose=0    
    arg=(param.get_upsilon_gt(),
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g,verbose)          
    
    cons = (
        #{'type':'ineq','fun': c.constr_hv_posit},
        #{'type':'ineq','fun': c.constr_sigv_posit},
        {'type':'ineq','fun': c.constr_Tvol_posit},
        {'type':'ineq','fun': c.constr_Tground_posit},
        #{'type':'ineq','fun': c.constr_gt12_inf1},
        #{'type':'ineq','fun': c.constr_gt13_inf1},
        #{'type':'ineq','fun': c.constr_gt23_inf1},
        #{'type':'ineq','fun': c.constr_gt12_posit},
        #{'type':'ineq','fun': c.constr_gt13_posit},
        #{'type':'ineq','fun': c.constr_gt23_posit},
           )
    """
    bds = [(0,None) for i in range(3)]+\
          [(0,None) for i in range(6)]+\
          [(0,None) for i in range(3)]+\
          [(0,None) for i in range(6)]+\
          [(0,0.1)]+\
          [(20,param.get_hamb_min())]+\
          [(0,1)]*3  
    """
    m = 1e6
    bds = [(0,m) for i in range(3)]+\
          [(0,m) for i in range(6)]+\
          [(0,m) for i in range(3)]+\
          [(0,m) for i in range(6)]+\
          [(20,param.get_hamb_min())]+\
          [(0,1)]+\
          [(0,1)]*3
    #mm.plot_analyse_mlogV_zg_connu_noapriori(param)
    
    k1 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    k2 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    eps_vol = 0.1
    eps_gro = 0.1
    Tvoln = param.T_vol + eps_vol*k1.dot(k1.T.conj())                       
    Tgroundn = param.T_ground + eps_gro*k2.dot(k2.T.conj())                       
    
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))
    dX = 0.01*np.sqrt(vec_bcr)
    
    X0 = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
                         mb.get_vecTm_from_Tm(Tgroundn),
                         np.array([param.h_v+dX[18]]),
                         np.array([param.sigma_v+dX[19]]),
                         0.95*np.ones(3)))

   
    eps = 1e-12
    MAXITER = 200
    
    X_SLSQP = opti.minimize(fun=e.mlogV_zg_known_noconstraint,
                      method='SLSQP',
                      x0=X0,
                      constraints=cons,
                      bounds=bds,
                      options={'ftol':eps,'disp':1,'iprint':2,'maxiter':MAXITER },
                      args=arg)
    
    """
    X_DE = opti.differential_evolution(func=e.mlogV_zg_known_noconstraint,
                                       bounds=bds,
                                       args=arg,maxiter=MAXITER,    
                                       disp=True)
    """
    """
    X_COBYLA = opti.minimize(fun=e.mlogV_zg_known_noconstraint,
                      method='COBYLA',
                      x0=X0,
                      constraints=cons,
                      options={'tol':1e-2,'disp':1,'iprint':2,'maxiter':MAXITER },
                      args=arg)
    """

    
    Xopt = X_SLSQP['x'][0] #Mdofi Capdessus : X['x'] contient [x,list_x,list_fx]
    mat_x = np.array(X_SLSQP['x'][1])
    vec_fx = np.array(X_SLSQP['x'][2])
    vec_Tvol = Xopt[:9]
    Tvol = mb.get_Tm_from_vecTm(vec_Tvol)
    vec_Tground = Xopt[9:18]
    Tground = mb.get_Tm_from_vecTm(vec_Tground)
    hv = Xopt[18]
    sigv = Xopt[19]
    Ngt = mb.get_Nb_from_Na(param.Na)
    
    if 20+Ngt != len(Xopt):
        print 'estim_mlogV_zg_known: pb taille Xopt'
    else:        
        vec_gt = Xopt[20:20+Ngt]

    
    print 'hv={0} hvvrai={1}'.format(hv,param.h_v)
    print 'sigv={0} sigvvvrai={1}'.format(sigv,param.sigma_v)
    print 'gt12={0} gt12vrai={1}'.format(vec_gt[0],param.get_gtlist()[0])
    print 'gt13={0} gt13vrai={1}'.format(vec_gt[1],param.get_gtlist()[1])
    print 'gt23={0} gt23vrai={1}'.format(vec_gt[2],param.get_gtlist()[2])
    print '------ Tvol ------ '
    bl.printm(Tvol)
    print '------ Tvolvrai ------ '
    bl.printm(param.T_vol)
    print '------ Tground ------ '
    bl.printm(Tground)
    print '------ Tgroundvrai ------ '
    bl.printm(param.T_ground)
    #print 'tvol1,tvol2,tvol3,tvol4,tvol5,tvol6,tvol7,tvol8,tvol9,tgro1,tgro2,tgro3,tgro4,tgro5,tgro6,tgro7,tgro8,tgro9,hv,sigv,gt12,gt13,gt23'
    #Erreur sur hv,sigv,gt12,gt13,gt23        
    vec_err_hv = np.abs(mat_x[:,18]-param.h_v)
    vec_err_sigv = np.abs(mat_x[:,19]-param.sigma_v)
    vec_err_gt12 = np.abs(mat_x[:,20]-param.gammat[0,1])
    vec_err_gt13 = np.abs(mat_x[:,21]-param.gammat[0,2])
    vec_err_gt23 = np.abs(mat_x[:,22]-param.gammat[1,2])
    mat_err1 = np.hstack((vec_err_hv[:,None],vec_err_sigv[:,None],
                         vec_err_gt12[:,None],vec_err_gt13[:,None],
                         vec_err_gt23[:,None]))
    name_err1=['hv','sigv','gt12','gt13','gt23']                     
    pt.pplot(range(len(vec_err_hv)),mat_err1,names=name_err1,
             yscale='log')
    
    #Erreur sur Tvol (1,2,3,4,5,6,7,8,9)    
    vec_Tvol = mb.get_vecTm_from_Tm(param.T_vol)
    vec_err_tvol1 = np.abs(mat_x[:,0]-vec_Tvol[0])
    vec_err_tvol2 = np.abs(mat_x[:,1]-vec_Tvol[1])
    vec_err_tvol3 = np.abs(mat_x[:,2]-vec_Tvol[2])
    vec_err_tvol4 = np.abs(mat_x[:,3]-vec_Tvol[3])
    vec_err_tvol5 = np.abs(mat_x[:,4]-vec_Tvol[4])
    vec_err_tvol6 = np.abs(mat_x[:,5]-vec_Tvol[5])
    vec_err_tvol7 = np.abs(mat_x[:,6]-vec_Tvol[6])
    vec_err_tvol8 = np.abs(mat_x[:,7]-vec_Tvol[7])
    vec_err_tvol9 = np.abs(mat_x[:,8]-vec_Tvol[8])
    
    mat_err2 = np.hstack((vec_err_tvol1[:,None],vec_err_tvol2[:,None],vec_err_tvol3[:,None],
                          vec_err_tvol4[:,None],vec_err_tvol5[:,None],vec_err_tvol6[:,None],
                          vec_err_tvol7[:,None],vec_err_tvol8[:,None],vec_err_tvol9[:,None],))
                          
    name_err2=['tvol1','tvol2','tvol3',
               'tvol4','tvol5','tvol6',
               'tvol7','tvol8','tvol9']
               
    pt.pplot(range(len(vec_err_tvol1)),mat_err2,names=name_err2,
             yscale='log')

    #Erreur sur Tgro (1,2,3,4,5,6,7,8,9)    
    vec_Tgro = mb.get_vecTm_from_Tm(param.T_ground)
    vec_err_tgro1 = np.abs(mat_x[:,9]-vec_Tgro[0])
    vec_err_tgro2 = np.abs(mat_x[:,10]-vec_Tgro[1])
    vec_err_tgro3 = np.abs(mat_x[:,11]-vec_Tgro[2])
    vec_err_tgro4 = np.abs(mat_x[:,12]-vec_Tgro[3])
    vec_err_tgro5 = np.abs(mat_x[:,13]-vec_Tgro[4])
    vec_err_tgro6 = np.abs(mat_x[:,14]-vec_Tgro[5])
    vec_err_tgro7 = np.abs(mat_x[:,15]-vec_Tgro[6])
    vec_err_tgro8 = np.abs(mat_x[:,16]-vec_Tgro[7])
    vec_err_tgro9 = np.abs(mat_x[:,17]-vec_Tgro[8])
    
    mat_err3 = np.hstack((vec_err_tgro1[:,None],vec_err_tgro2[:,None],vec_err_tgro3[:,None],
                          vec_err_tgro4[:,None],vec_err_tgro5[:,None],vec_err_tgro6[:,None],
                          vec_err_tgro7[:,None],vec_err_tgro8[:,None],vec_err_tgro9[:,None],))
                          
    name_err3 = ['tgro1','tgro2','tgro3',
                'tgro4','tgro5','tgro6',
                'tgro7','tgro8','tgro9']
               
    pt.pplot(range(len(vec_err_tgro1)),mat_err3,names=name_err3,
             yscale='log')


    #pt.plot_converg(vec_fx,crit_vrai=Crit_vrai,mode='diff')
    pt.plot_converg(vec_fx)
    #Evoution de |f(Xk+1)-f(Xk)|
    diff_crit = np.abs(np.diff(vec_fx))
    pt.plot_converg(diff_crit);plt.title('diff_crit |f(Xn+1)-f(Xn)|')
    #diff_norm_crit = np.abs(np.diff(vec_fx))/vec_fx[:-1]
    diff_norm_crit = np.abs(np.diff(vec_fx))/vec_fx[1:]
    
    pt.plot_converg(diff_norm_crit);plt.title('diff_norm_crit |f(Xn+1)-f(Xn)|/f(Xn)')
    
