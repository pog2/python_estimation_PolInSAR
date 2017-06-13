# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:00:15 2016

@author: Pierre
Etude de la fonction mlogV_zg_connu()
"""
from __future__ import division
from matplotlib.colors import LogNorm
import plot_tomo as pt
import sys
if '../Polinsar/' not in sys.path:
    sys.path.append('../Polinsar/')
if '../stat/' not in sys.path:
    sys.path.append('../stat/')
import numpy as np
import numpy.linalg as npl
import matplotlib.mlab as mat
import matplotlib.pyplot as plt
import zone_analyse as za
import stat_SAR as st
import RVoG_MB as mb
import basic_lib as bl
import os 
import load_param as lp
import estimation as e 
import scipy.optimize as opti
import contraintes as c
import pdb
import tomosar_synth_v5 as tom

def printX(X):
    np.set_printoptions(precision=3,linewidth=400)    
    print X

def histogramme_autour_de_la_valeur_vraie(param):

    """ Recherche du min : histogramme de Tirages autour de la valeur vraie"""    
    X=np.concatenate((mb.get_vecTm_from_Tm(param.T_vol),
                     mb.get_vecTm_from_Tm(param.T_ground),
                     np.array([param.h_v]),
                     np.array([param.sigma_v]),
                     param.get_gtlist()))
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))
    P = 5000
    Crit = np.zeros(P)
    
    Crit_vrai = e.mlogV_zg_known_noconstraint(X,param.get_upsilon_gt(),
                               param.get_kzlist(),
                               np.cos(param.theta),
                               param.Na,param.N,
                               param.z_g)
    for i in range(P):        
        dX = 0.001*np.sqrt(vec_bcr)*np.random.randn(23)
        Crit[i] = e.mlogV_zg_known_noconstraint(X+dX,param.get_upsilon_gt(),
                                   param.get_kzlist(),
                                   np.cos(param.theta),
                                   param.Na,param.N,
                                   param.z_g)

    
    plt.figure()
    bl.hhist(Crit[np.isfinite(Crit)],title='mlogV_zg_known')
    plt.hold(True)
    plt.axvline(Crit_vrai)

def recherche_min_optimize_noise(param):
    param = lp.load_param('DB_1')    
    param.gammat = 0.95*np.ones((3,3))
    param.N = 1e2
    
    """ Recherche du min (opti.fmin) : algo d'opti en presence de bruit""" 
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))
    dX = np.sqrt(vec_bcr)*1
    #dX[:-4] = np.zeros(19)
        
    Xvrai = np.concatenate((mb.get_vecTm_from_Tm(param.T_vol),
                         mb.get_vecTm_from_Tm(param.T_ground),
                         np.array([param.h_v]),
                         np.array([param.sigma_v]),
                         param.get_gtlist()))

    var1 = np.sum(vec_bcr[:9])
    var2 = np.sum(vec_bcr[9:18])    
    #k1 = 0.5*np.sqrt(var1)*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    k1 = 0.5*0.1*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    #k2 = 0.5*np.sqrt(var2)*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    k2 = 0.5*0.1*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    Tvoln = param.T_vol + k1.dot(k1.T.conj())                       
    Tgroundn = param.T_ground + k2.dot(k2.T.conj())                       
    
    X0 = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
                         mb.get_vecTm_from_Tm(Tgroundn),
                         np.array([param.h_v+dX[18]]),
                         np.array([param.sigma_v+dX[19]]),
                         0.8*np.ones(3)))
    
    #X0 = Xvrai
    data = tom.TomoSARDataSet_synth(param)
    Ups_n = data.get_covar_rect(param,param.N)
    #Ups_n = param.get_upsilon_gt()
    
    verbose=1
    #arg[0]: Upsilon bruité
    arg=(Ups_n,
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g,verbose)   
         
    bds = [(0,None) for i in range(3)]+\
          [(None,None) for i in range(6)]+\
          [(0,None) for i in range(3)]+\
          [(None,None) for i in range(6)]+\
          [(0,None)]*2+\
          [(0,1)]*3
          
    X = opti.minimize(fun=e.mlogV_zg_known,
                      method='SLSQP',
                      x0=X0,
                      options={'xtol':1e-10,'ftol':1e-10,
                               'disp':1,'iprint':2,
                                'maxiter':200},                     
                      bounds=bds,
                      args=arg) 
                      
    
    Crit_vrai = e.mlogV_zg_known(Xvrai,*arg)                            
    print '|C-Cvrai|={0}'.format(np.abs(X['fun']-Crit_vrai))    
    Xopt = X['x'][0] #Mdofi Capdessus : X['x'] contient [x,list_x,list_fx]
    
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

    #Erreur sur hv,sigv,gt12,gt13,gt23        
    vec_err_hv = np.abs(np.array([X['x'][1][i][18] for i in range(len(X['x'][1]))])-param.h_v)#-param.h_v)
    vec_err_sigv = np.abs(np.array([X['x'][1][i][19] for i in range(len(X['x'][1]))])-param.sigma_v)#-param.sigma_v)
    vec_err_gt12 = np.abs(np.array([X['x'][1][i][20] for i in range(len(X['x'][1]))])-param.gammat[0,1])#-param.gammat[0,1])
    vec_err_gt13 = np.abs(np.array([X['x'][1][i][21] for i in range(len(X['x'][1]))])-param.gammat[0,2])#-param.gammat[0,2])
    vec_err_gt23 = np.abs(np.array([X['x'][1][i][22] for i in range(len(X['x'][1]))])-param.gammat[1,2])#-param.gammat[1,2])
    mat_err1 = np.hstack((vec_err_hv[:,None],vec_err_sigv[:,None],
                         vec_err_gt12[:,None],vec_err_gt13[:,None],
                         vec_err_gt23[:,None]))
    name_err1=['hv','sigv','gt12','gt13','gt23']                     
    pt.pplot(range(len(vec_err_hv)),mat_err1,names=name_err1,
             yscale='log')
    
    #Erreur sur Tvol (1,2,3,4,5,6,7,8,9)    
    vec_Tvol = mb.get_vecTm_from_Tm(param.T_vol)
    vec_err_tvol1 = np.abs(np.array([X['x'][1][i][0] for i in range(len(X['x'][1]))])-vec_Tvol[0])
    vec_err_tvol2 = np.abs(np.array([X['x'][1][i][1] for i in range(len(X['x'][1]))])-vec_Tvol[1])
    vec_err_tvol3 = np.abs(np.array([X['x'][1][i][2] for i in range(len(X['x'][1]))])-vec_Tvol[2])
    vec_err_tvol4 = np.abs(np.array([X['x'][1][i][3] for i in range(len(X['x'][1]))])-vec_Tvol[3])
    vec_err_tvol5 = np.abs(np.array([X['x'][1][i][4] for i in range(len(X['x'][1]))])-vec_Tvol[4])
    vec_err_tvol6 = np.abs(np.array([X['x'][1][i][5] for i in range(len(X['x'][1]))])-vec_Tvol[5])
    vec_err_tvol7 = np.abs(np.array([X['x'][1][i][6] for i in range(len(X['x'][1]))])-vec_Tvol[6])
    vec_err_tvol8 = np.abs(np.array([X['x'][1][i][7] for i in range(len(X['x'][1]))])-vec_Tvol[7])
    vec_err_tvol9 = np.abs(np.array([X['x'][1][i][8] for i in range(len(X['x'][1]))])-vec_Tvol[8])
    
    mat_err2 = np.hstack((vec_err_tvol1[:,None],vec_err_tvol2[:,None],vec_err_tvol3[:,None],
                          vec_err_tvol4[:,None],vec_err_tvol5[:,None],vec_err_tvol6[:,None],
                          vec_err_tvol7[:,None],vec_err_tvol8[:,None],vec_err_tvol9[:,None],))
                          
    name_err2=['tvol1','tvol2','tvol3',
               'tvol4','tvol5','tvol6',
               'tvol7','tvol8','tvol9']
               
    pt.pplot(range(len(vec_err_tvol1)),mat_err2,names=name_err2,yscale='log')

    #Erreur sur Tgro (1,2,3,4,5,6,7,8,9)    
    vec_Tgro = mb.get_vecTm_from_Tm(param.T_ground)
    vec_err_tgro1 = np.abs(np.array([X['x'][1][i][9] for i in range(len(X['x'][1]))])-vec_Tgro[0])
    vec_err_tgro2 = np.abs(np.array([X['x'][1][i][10] for i in range(len(X['x'][1]))])-vec_Tgro[1])
    vec_err_tgro3 = np.abs(np.array([X['x'][1][i][11] for i in range(len(X['x'][1]))])-vec_Tgro[2])
    vec_err_tgro4 = np.abs(np.array([X['x'][1][i][12] for i in range(len(X['x'][1]))])-vec_Tgro[3])
    vec_err_tgro5 = np.abs(np.array([X['x'][1][i][13] for i in range(len(X['x'][1]))])-vec_Tgro[4])
    vec_err_tgro6 = np.abs(np.array([X['x'][1][i][14] for i in range(len(X['x'][1]))])-vec_Tgro[5])
    vec_err_tgro7 = np.abs(np.array([X['x'][1][i][15] for i in range(len(X['x'][1]))])-vec_Tgro[6])
    vec_err_tgro8 = np.abs(np.array([X['x'][1][i][16] for i in range(len(X['x'][1]))])-vec_Tgro[7])
    vec_err_tgro9 = np.abs(np.array([X['x'][1][i][17] for i in range(len(X['x'][1]))])-vec_Tgro[8])
    
    mat_err3 = np.hstack((vec_err_tgro1[:,None],vec_err_tgro2[:,None],vec_err_tgro3[:,None],
                          vec_err_tgro4[:,None],vec_err_tgro5[:,None],vec_err_tgro6[:,None],
                          vec_err_tgro7[:,None],vec_err_tgro8[:,None],vec_err_tgro9[:,None],))
                          
    name_err3 = ['tgro1','tgro2','tgro3',
                'tgro4','tgro5','tgro6',
                'tgro7','tgro8','tgro9']
               
    pt.pplot(range(len(vec_err_tgro1)),mat_err3,names=name_err3,yscale='log')


    vec_crit = np.array([X['x'][2][i] for i in range(len(X['x'][2]))])
    pt.plot_converg(vec_crit,crit_vrai=Crit_vrai,mode='diff',show_inf=True)
    pt.plot_converg(vec_crit)        
    
    
def recherche_min_optimize_nonoise(param):
   
    """ Recherche du min (opti.fmin) : algo d'opti dans le cas non bruite"""        
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))
    dX = 0.1*np.sqrt(vec_bcr)
        
    Xvrai = np.concatenate((mb.get_vecTm_from_Tm(param.T_vol),
                         mb.get_vecTm_from_Tm(param.T_ground),
                         np.array([param.h_v]),
                         np.array([param.sigma_v]),
                         param.get_gtlist()))
 
    k1 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    k2 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    eps_vol = 0
    eps_gro = 0
    Tvoln = param.T_vol + eps_vol*k1.dot(k1.T.conj())                       
    Tgroundn = param.T_ground + eps_gro*k2.dot(k2.T.conj())                       
    
    X0 = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
                         mb.get_vecTm_from_Tm(Tgroundn),
                         np.array([param.h_v+dX[18]]),
                         np.array([param.sigma_v+dX[19]]),
                         0.8*np.ones(3)))

    """
    cons = ({'type':'ineq','fun': c.constr_hv_posit},
            {'type':'ineq','fun': c.constr_sigv_posit},
            {'type':'ineq','fun': c.constr_Tvol_posit},
            {'type':'ineq','fun': c.constr_Tground_posit},
            {'type':'ineq','fun': c.constr_gt12_inf1},
            {'type':'ineq','fun': c.constr_gt13_inf1},
            {'type':'ineq','fun': c.constr_gt23_inf1},
            {'type':'ineq','fun': c.constr_gt12_posit},
            {'type':'ineq','fun': c.constr_gt13_posit},
            {'type':'ineq','fun': c.constr_gt23_posit},
           )
    """
    #(Tvol,Tground,hv,sig,gt12,gt13,gt23)
    bds = [(0,None) for i in range(3)]+\
          [(None,None) for i in range(6)]+\
          [(0,None) for i in range(3)]+\
          [(None,None) for i in range(6)]+\
          [(0,None)]*2+\
          [(0,1)]*3
          
    verbose = 0
    arg=(param.get_upsilon_gt(),
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g,verbose)            
         
    eps64 = np.finfo(np.float64).eps
    eps32 = np.finfo(np.float32).eps
    eps = np.finfo(np.float).eps
    eps0 =np.finfo(np.floating).eps
    
    eps = 1e-12
    X = opti.minimize(fun=e.mlogV_zg_known,
                      method='SLSQP',
                      x0=X0,
                      #callback = printX,
                      #bounds=bds,
                      options={'ftol':eps,'disp':1,'iprint':2,'maxiter':200},
                      args=arg) 
                          
    Crit_vrai = e.mlogV_zg_known(Xvrai,*arg)                            
    #print '|C-Cvrai|={0}'.format(np.abs(X['fun']-Crit_vrai))    
    Xopt = X['x'][0] #Mdofi Capdessus : X['x'] contient [x,list_x,list_fx]
    mat_x = np.array(X['x'][1])
    vec_fx = np.array(X['x'][2])
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
    
    #sys.exit()
    #Evolution de l'erreur sur hv en fonction de eps_vol et esp_gro
    
    
    verbose = 0                     
    arg=(param.get_upsilon_gt(),
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g,verbose)   
                         
    Pinit = 20 #Nbre d'initialisation differentes
    Nb_eps_vol = 5
    Nb_eps_gro = 5
    tab_err_hv = np.zeros((Nb_eps_vol,Nb_eps_gro,Pinit))
    mat_EQM_hv = np.zeros((Nb_eps_vol,Nb_eps_gro))
    mat_var_hv = np.zeros((Nb_eps_vol,Nb_eps_gro))
    mat_Preussi = np.zeros((Nb_eps_vol,Nb_eps_gro),dtype=int)

    mat_estim_err_hv = np.zeros((Nb_eps_vol,Nb_eps_gro),dtype=object)
    mat_estim_mean_err_hv = np.zeros((Nb_eps_vol,Nb_eps_gro),dtype=object)
    mat_estim_std_err_hv = np.zeros((Nb_eps_vol,Nb_eps_gro),dtype=object)
    
    mat_estim_difffx=np.zeros((Nb_eps_vol,Nb_eps_gro),dtype=object)
    mat_estim_mean_difffx=np.zeros((Nb_eps_vol,Nb_eps_gro),dtype=object)
    mat_estim_std_difffx=np.zeros((Nb_eps_vol,Nb_eps_gro),dtype=object)        
    
    mat_status = np.zeros((Nb_eps_vol,Nb_eps_gro))
    vec_eps_vol = np.logspace(-2,1,Nb_eps_vol)
    vec_eps_gro = np.logspace(-2,3,Nb_eps_gro)
    
    list_xnoconverg = np.ndarray((Nb_eps_vol,Nb_eps_gro),dtype=object)
    list_fxnoconverg = np.ndarray((Nb_eps_vol,Nb_eps_gro),dtype=object)
    list_xconverg = np.ndarray((Nb_eps_vol,Nb_eps_gro),dtype=object)
    list_fxconverg = np.ndarray((Nb_eps_vol,Nb_eps_gro),dtype=object)    
    list_difffxconverg = np.ndarray((Nb_eps_vol,Nb_eps_gro),dtype=object)
    
    for i,eps_vol in enumerate(vec_eps_vol):
        for j,eps_gro in enumerate(vec_eps_gro):                        
            list_xnoconverg[i,j] = []
            list_fxnoconverg[i,j] = []
            list_xconverg[i,j] = []
            list_fxconverg[i,j] = []     
            list_difffxconverg[i,j] = []
            for k in range(Pinit):
                k1 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))
                k2 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))
                Tvoln = param.T_vol + eps_vol*k1.dot(k1.T.conj())                       
                Tgroundn = param.T_ground + eps_gro*k2.dot(k2.T.conj())                       
                
                X0 = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
                                     mb.get_vecTm_from_Tm(Tgroundn),
                                     np.array([param.h_v+dX[18]]),
                                     np.array([param.sigma_v+dX[19]]),
                                     0.8*np.ones(3)))    
                                     
                X = opti.minimize(fun=e.mlogV_zg_known,
                      method='SLSQP',
                      x0=X0,
                      options={'ftol':eps,'disp':1,'iprint':2,'maxiter':200},
                      args=arg) 
            
                if (X['status'] == 0 or X['status'] == 9)  and X['fun'] != 1e14:                    
                    Xopt = X['x'][0] #Mdofi Capdessus : X['x'] contient [x,list_x,list_fx]
                    hv = Xopt[18]
                    tab_err_hv[i,j,k] = np.abs(hv - param.h_v)
                    mat_Preussi[i,j] = mat_Preussi[i,j] + 1 
                    list_xconverg[i,j].append(np.abs(X['x'][1]-Xvrai))                    
                    list_fxconverg[i,j].append(X['x'][2])
                    list_difffxconverg[i,j].append(np.abs(np.diff(X['x'][2])))
                    
                else:
                    Xopt = X['x'][0] #Mdofi Capdessus : X['x'] contient [x,list_x,list_fx]           
                    list_xnoconverg[i,j].append(np.abs(X['x'][1]-Xvrai))
                    list_fxnoconverg[i,j].append(X['x'][2])
                    hv = Xopt[18]
                    tab_err_hv[i,j,k] = np.nan
                                                        
                #np.abs(hv-param.hv)**2
            
            if mat_Preussi[i,j] != 0 :                    
                min_iter=np.min(np.array([len(list_xconverg[i,j][k][:,18]) for k in range(mat_Preussi[i,j])]))
                mat_estim_err_hv[i,j] = np.zeros((mat_Preussi[i,j],min_iter))
                mat_estim_err_hv[i,j] = np.array([list_xconverg[i,j][k][:min_iter,18] for k in range(mat_Preussi[i,j])])
                mat_estim_mean_err_hv[i,j] = np.zeros(min_iter)
                mat_estim_mean_err_hv[i,j] = np.mean(mat_estim_err_hv[i,j],axis=0)
                mat_estim_std_err_hv[i,j] = np.std(mat_estim_err_hv[i,j],axis=0)
                
                mat_estim_difffx[i,j] = np.zeros((mat_Preussi[i,j],min_iter-1))
                mat_estim_difffx[i,j] = np.array([np.abs(np.diff(list_fxconverg[i,j][k][:min_iter])) for k in range(mat_Preussi[i,j])])
                mat_estim_mean_difffx[i,j] = np.zeros(min_iter)
                mat_estim_mean_difffx[i,j] = np.mean(mat_estim_difffx[i,j],axis=0)
                mat_estim_std_difffx[i,j] = np.std(mat_estim_difffx[i,j],axis=0)
 
            mat_status[i,j] = X['status']      
            mat_EQM_hv[i,j] = np.nanmean(tab_err_hv[i,j,:].ravel()**2)
            mat_var_hv[i,j] = np.nanvar(tab_err_hv[i,j,:])
            
    
    
    """DEvelopement deans le main """
    
    
def plot_analyse_mlogV_zg_connu(param,Xinit=None):
    """' Plot du critère mlogV_zg_connu en faisant varier chaque paramètres.
    Les autres étant fixés à la valeur vraie"""
    
    plt_init = 0
    list_name = ['tvol1','tvol2','tvol3',
                 'tvol4','tvol5','tvol6',
                 'tvol7','tvol8','tvol9',
                 'tground1','tground2','tground3',
                 'tground4','tground5','tground6',
                 'tground7','tground8','tground9',
                 'hv','sigv','gt12','gt13','gt23',
                 ]
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))                 
    dX = np.sqrt(vec_bcr)
    
    var1 = np.sum(vec_bcr[:9])
    var2 = np.sum(vec_bcr[9:18])    
    k1 = 0.5*np.sqrt(var1)*(np.random.randn(3,1)+1j*np.random.randn(3,1))                         
    k2 = 0.5*np.sqrt(var2)*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    
    Tvoln = param.T_vol + k1.dot(k1.T.conj())                       
    Tgroundn = param.T_ground + k2.dot(k2.T.conj())                       
    
 
    #0.5*np.ones(3)
    
    X0 = np.concatenate((mb.get_vecTm_from_Tm(param.T_vol),
                     mb.get_vecTm_from_Tm(param.T_ground),
                     np.array([param.h_v]),
                     np.array([param.sigma_v]),
                     param.get_gtlist()))
#    Xinit = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
#                     mb.get_vecTm_from_Tm(Tgroundn),
#                     np.array([param.h_v+dX[18]]),
#                     np.array([param.sigma_v+dX[19]]),
#                     param.get_gtlist()+dX[-3:]))   
#    Xinit = X0+dX
#     Xinit[-3:] = 0.5*np.ones(3)
    if Xinit is None:
        
        Xinit = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
                                mb.get_vecTm_from_Tm(Tgroundn),
                                np.array([param.h_v+dX[18]]),
                                np.array([param.sigma_v+dX[19]]),
                                0.5*np.ones(3)))                     
    
    Xinit = X0+dX
                                
    arg=(param.get_upsilon_gt(),
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g)      
         
    Critinit = e.mlogV_zg_known(Xinit,*arg)
    
    Npt = 500
    Xmin = np.amin(np.hstack((X0[:,None]-dX[:,None],Xinit[:,None])),axis=1)
    Xmax = np.amax(np.hstack((X0[:,None]+dX[:,None],Xinit[:,None])),axis=1)
    #pdb.set_trace()
#    Xmin = X0[:,None]-dX[:,None]
#    Xmax = X0[:,None]+dX[:,None]
    
    #On incorpore les contraintes (Tvol1,2,3>=0,Tground1,2,3>=0,sigv>0,gt€[0,1])
    
#    Xmin2[[range(3)+range(9,12)+range(19,23)]] = np.zeros((6,1))
#    Xmax2[[range(20,23)]] = np.ones((6,1))#pas de sup pour Tvol,Tgro et sigv
#    
    #Matrice contenant sur la ligne i la variation du paramètre i (linspace)
    mat_X = np.array([np.linspace(xmin,xmax,Npt) for xmin,xmax in zip(Xmin,Xmax)])    
    mat_X0 = X0[:,None].dot(np.ones((1,Npt)))
    #Dans le ieme element de la liste seul la ieme ligne varie
    list_mat_X = np.zeros((len(X0),mat_X.shape[0],mat_X.shape[1]))
    mat_crit = np.zeros((len(X0),Npt))
    vec_crit_init = np.zeros(len(X0)) #critere init (ie compos=Xinit[i], le reste = val vraie)
    #mat_crit_init : i colonne = X0 sauf la i ligne qui est Xinit[i]
    mat_crit_init = mat_X0.copy()
    mat_crit_init[range(len(X0)),range(len(X0))] = Xinit
    for i in range(len(X0)):
        list_mat_X[i,:,:] = mat_X0
        list_mat_X[i,i,:] = mat_X[i,:]
    
    for i in range(len(X0)):
        vec_crit_init[i] = e.mlogV_zg_known(mat_crit_init[:,i],*arg)
        for j in range(Npt):            
            X = list_mat_X[i,:,j]        
            #if i == 0 : print str(i),['{:3.3f}'.format(X[i]) for i in range(len(X))]          
            mat_crit[i,j] = e.mlogV_zg_known(X,*arg)
    
    
    
    #Pour éviter que les echelles des plots soient biaisées
    #on ne trace pas les 1e14 (remplacement par nan)
    idx = np.where(mat_crit>=1e14)
    mat_crit[idx] = np.nan
    #Subplot 3x3 -> Tvol
    f_Tvol = plt.figure()
    for i in range(9):
        f_Tvol.add_subplot(3,3,i+1)
        plt.plot(mat_X[i,:],mat_crit[i,:])
        if plt_init : plt.plot(Xinit[i],vec_crit_init[i],'or')
        plt.axvline(X0[i],ymin=0,ymax=0.5,color='red',ls='--')        
        if i<3: 
            #plt.ylim((3.5e3,4e3))
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')
            ymin = np.nanmin(mat_crit[i,:])-0.05*np.nanmin(mat_crit[i,:])
            ymax = np.nanmin(mat_crit[i,:])+0.05*np.nanmin(mat_crit[i,:])
            plt.ylim((ymin,ymax))
            plt.grid()
        else:
            #pdb.set_trace()
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')            
            ymin = 3539.8
            ymax = 3540.6
            plt.ylim((ymin,ymax))
            plt.grid()
        plt.xlabel(list_name[i])        
        
    #Subplot 3x3 -> Tgro
    f_Tground = plt.figure()
    for i in range(9):
        f_Tground.add_subplot(3,3,i+1)
        plt.plot(mat_X[i+9,:],mat_crit[i+9,:])
        plt.axvline(X0[i+9],ymin=0,ymax=0.5,color='red',ls='--')
        if plt_init : plt.plot(Xinit[i+9],vec_crit_init[i+9],'or')
        if i<3 : 
            #plt.ylim((3.5e3,4e3))
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')
            ymin = np.nanmin(mat_crit[i,:])-0.05*np.nanmin(mat_crit[i,:])
            ymax = np.nanmin(mat_crit[i,:])+0.05*np.nanmin(mat_crit[i,:])
            plt.ylim((ymin,ymax))
            plt.grid()
        else:
            #pdb.set_trace()
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')            
            ymin = 3539.8
            ymax = 3540.6
            plt.ylim((ymin,ymax))
            plt.grid()
        plt.xlabel(list_name[i+9])        
        
    #Subplot 3x3 -> (hv,sigv,gt12,gt13,gt23)
    f_last = plt.figure()
    for i in range(5):
        f_last.add_subplot(2,3,i+1)
        plt.plot(mat_X[i+18,:],mat_crit[i+18,:])
        if plt_init : plt.plot(Xinit[i+18],vec_crit_init[i+18],'or')
        if i<2 : 
            plt.yscale('linear')
            plt.ylim((3.5e3,4e3))
            
        elif i==2:
            plt.ylim((3.5e3,3.6e3))
        elif i==3:
            plt.ylim((3.535e3,3.545e3))            
        elif i==4:
            plt.ylim(3.539e3,3.542e3)            
        else:
            plt.yscale('linear')  
            plt.ylim((3.54e3,3.6e3))
        plt.grid(True)
        plt.ylabel('mlogV(linear)')            
        plt.axvline(X0[i+18],ymin=0,ymax=0.5,color='red',ls='--')
        plt.xlabel(list_name[i+18])        
    print vec_crit_init


def plot_analyse_mlogV_zg_connu_noapriori(param,Xinit=None):
    """' Plot du critère mlogV_zg_connu-mlogV_zg_connu(Xvrai)
    en faisant varier chaque paramètres.
    Les autres étant fixés à la valeur vraie"""
    
    plt_init = 0
    list_name = ['tvol1','tvol2','tvol3',
                 'tvol4','tvol5','tvol6',
                 'tvol7','tvol8','tvol9',
                 'tground1','tground2','tground3',
                 'tground4','tground5','tground6',
                 'tground7','tground8','tground9',
                 'hv','sigv','gt12','gt13','gt23',
                 ]
                 
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))                 
    dX = np.sqrt(vec_bcr)
    #dX[18] = 10
    
    var1 = np.sum(vec_bcr[:9])
    var2 = np.sum(vec_bcr[9:18])    
    k1 = 0.5*np.sqrt(var1)*(np.random.randn(3,1)+1j*np.random.randn(3,1))                         
    k2 = 0.5*np.sqrt(var2)*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    
    eps_vol = 0.01
    eps_gro = 2
    Tvoln = param.T_vol + eps_vol*k1.dot(k1.T.conj())                       
    Tgroundn = param.T_ground + eps_gro*k2.dot(k2.T.conj())                       
        
    X0 = np.concatenate((mb.get_vecTm_from_Tm(param.T_vol),
                     mb.get_vecTm_from_Tm(param.T_ground),
                     np.array([param.h_v]),
                     np.array([param.sigma_v]),
                     param.get_gtlist()))

    if Xinit is None:
        Xinit = X0+dX        
                                    
    arg=(param.get_upsilon_gt(),
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g)      
         
    Critinit = e.mlogV_zg_known(Xinit,*arg)
    CritVrai = e.mlogV_zg_known(X0,*arg)
    Npt = 100
    Xmin = np.amin(np.hstack((X0[:,None]-dX[:,None],Xinit[:,None])),axis=1)
    Xmax = np.amax(np.hstack((X0[:,None]+dX[:,None],Xinit[:,None])),axis=1)
    #pdb.set_trace()
#    Xmin = X0[:,None]-dX[:,None]
#    Xmax = X0[:,None]+dX[:,None]
    
    #On incorpore les contraintes (Tvol1,2,3>=0,Tground1,2,3>=0,sigv>0,gt€[0,1])
    
#    Xmin2[[range(3)+range(9,12)+range(19,23)]] = np.zeros((6,1))
#    Xmax2[[range(20,23)]] = np.ones((6,1))#pas de sup pour Tvol,Tgro et sigv
#    
    #Matrice contenant sur la ligne i la variation du paramètre i (linspace)
    mat_X = np.array([np.linspace(xmin,xmax,Npt) for xmin,xmax in zip(Xmin,Xmax)])    
    mat_X0 = X0[:,None].dot(np.ones((1,Npt)))
    #Dans le ieme element de la liste seul la ieme ligne varie
    list_mat_X = np.zeros((len(X0),mat_X.shape[0],mat_X.shape[1]))
    mat_crit = np.zeros((len(X0),Npt))
    vec_crit_init = np.zeros(len(X0)) #critere init (ie compos=Xinit[i], le reste = val vraie)
    #mat_crit_init : i colonne = X0 sauf la i ligne qui est Xinit[i]
    mat_crit_init = mat_X0.copy()
    mat_crit_init[range(len(X0)),range(len(X0))] = Xinit
    for i in range(len(X0)):
        list_mat_X[i,:,:] = mat_X0
        list_mat_X[i,i,:] = mat_X[i,:]
    
    for i in range(len(X0)):
        vec_crit_init[i] = e.mlogV_zg_known(mat_crit_init[:,i],*arg)-CritVrai
        for j in range(Npt):            
            X = list_mat_X[i,:,j]        
            #if i == 0 : print str(i),['{:3.3f}'.format(X[i]) for i in range(len(X))]          
            mat_crit[i,j] = e.mlogV_zg_known(X,*arg)-CritVrai
            
    #Pour éviter que les echelles des plots soient biaisées
    #on ne trace pas les 1e14 (remplacement par nan)
    idx = np.where(mat_crit+CritVrai>=1e14)
    mat_crit[idx] = np.nan
    #Subplot 3x3 -> Tvol
    f_Tvol = plt.figure()
    for i in range(9):
        f_Tvol.add_subplot(3,3,i+1)
        plt.plot(mat_X[i,:],mat_crit[i,:])
        if plt_init : plt.plot(Xinit[i],vec_crit_init[i],'or')
        plt.axvline(X0[i],ymin=0,ymax=0.5,color='red',ls='--')        
        if i<3: 
            #plt.ylim((3.5e3,4e3))
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')
            #ymin = np.nanmin(mat_crit[i,:])-0.05*np.nanmin(mat_crit[i,:])
            #ymax = np.nanmin(mat_crit[i,:])+0.05*np.nanmin(mat_crit[i,:])
            #plt.ylim((ymin,ymax))
            plt.grid()
        else:
            #pdb.set_trace()
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')            
            #ymin = 3539.8
            #ymax = 3540.6
            #plt.ylim((ymin,ymax))
            plt.grid()
        plt.xlabel(list_name[i])        
        
    #Subplot 3x3 -> Tgro
    f_Tground = plt.figure()
    for i in range(9):
        f_Tground.add_subplot(3,3,i+1)
        plt.plot(mat_X[i+9,:],mat_crit[i+9,:])
        plt.axvline(X0[i+9],ymin=0,ymax=0.5,color='red',ls='--')
        if plt_init : plt.plot(Xinit[i+9],vec_crit_init[i+9],'or')
        if i<3 : 
            #plt.ylim((3.5e3,4e3))
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')
            #ymin = np.nanmin(mat_crit[i,:])-0.05*np.nanmin(mat_crit[i,:])
            #ymax = np.nanmin(mat_crit[i,:])+0.05*np.nanmin(mat_crit[i,:])
            #plt.ylim((ymin,ymax))
            plt.grid()
        else:
            #pdb.set_trace()
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')            
            #ymin = 3539.8
            #ymax = 3540.6
            #plt.ylim((ymin,ymax))
            plt.grid()
        plt.xlabel(list_name[i+9])        
        
    #Subplot 3x3 -> (hv,sigv,gt12,gt13,gt23)
    f_last = plt.figure()
    for i in range(5):
        f_last.add_subplot(2,3,i+1)
        plt.plot(mat_X[i+18,:],mat_crit[i+18,:])
        if plt_init : plt.plot(Xinit[i+18],vec_crit_init[i+18],'or')
        if i<2 : 
            plt.yscale('linear')
            #plt.ylim((3.5e3,4e3))
        else:
            plt.yscale('linear')  
            #plt.ylim((3.54e3,3.6e3))
        plt.grid(True)
        plt.ylabel('mlogV(linear)')            
        plt.axvline(X0[i+18],ymin=0,ymax=0.5,color='red',ls='--')
        plt.xlabel(list_name[i+18])        
    print vec_crit_init
    #pdb.set_trace()
    
def plot_analyse_mlogV_zg_connu_noise(param,Xinit=None,Xend=None,Ups_n=None):
    """' Plot du critère mlogV_zg_connu en faisant varier chaque paramètres.
    Les autres étant fixés à la valeur vraie"""
    
    plt_init = 1
    list_name = ['tvol1','tvol2','tvol3',
                 'tvol4','tvol5','tvol6',
                 'tvol7','tvol8','tvol9',
                 'tground1','tground2','tground3',
                 'tground4','tground5','tground6',
                 'tground7','tground8','tground9',
                 'hv','sigv','gt12','gt13','gt23',
                 ]
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))                 
    dX = np.sqrt(vec_bcr)
    
    var1 = np.sum(vec_bcr[:9])
    var2 = np.sum(vec_bcr[9:18])    
    k1 = 0.5*np.sqrt(var1)*(np.random.randn(3,1)+1j*np.random.randn(3,1))                         
    k2 = 0.5*np.sqrt(var2)*(np.random.randn(3,1)+1j*np.random.randn(3,1))
    
    Tvoln = param.T_vol + k1.dot(k1.T.conj())                       
    Tgroundn = param.T_ground + k2.dot(k2.T.conj())                       
    
 
    #0.5*np.ones(3)
    
    X0 = np.concatenate((mb.get_vecTm_from_Tm(param.T_vol),
                     mb.get_vecTm_from_Tm(param.T_ground),
                     np.array([param.h_v]),
                     np.array([param.sigma_v]),
                     param.get_gtlist()))
                     

    if Xinit is None:
        
        Xinit = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
                                mb.get_vecTm_from_Tm(Tgroundn),
                                np.array([param.h_v+dX[18]]),
                                np.array([param.sigma_v+dX[19]]),
                                0.5*np.ones(3)))                     
    if Xend is not None:
        plt_end = 1 
        
    if Ups_n is None:                                
        data = tom.TomoSARDataSet_synth(param)
        Ups_n = data.get_covar_rect(param,param.N)
        
    arg=(Ups_n,
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g)      
         
    Critinit = e.mlogV_zg_known(Xinit,*arg)
    
    Npt = 500
    Dplot = np.amax(np.hstack((np.abs(X0[:,None]-dX[:,None]),
                               np.abs(X0[:,None]-Xinit[:,None]))),axis=1)
    #Xmin = np.amin(np.hstack((X0[:,None]-dX[:,None],Xinit[:,None])),axis=1)
    #Xmax = np.amax(np.hstack((X0[:,None]+dX[:,None],Xinit[:,None])),axis=1)
    Xmin = X0[:,None] - Dplot[:,None]
    Xmax = X0[:,None] + Dplot[:,None]
    
    #Matrice contenant sur la ligne i la variation du paramètre i (linspace)
    mat_X = np.array([np.linspace(xmin,xmax,Npt) for xmin,xmax in zip(Xmin,Xmax)])    
    #mat_X0 contient le vec colonne de param vrai, dupliqué Npt fois
    mat_X0 = X0[:,None].dot(np.ones((1,Npt)))
    #Dans le ieme element de la liste seul la ieme ligne varie
    list_mat_X = np.zeros((len(X0),mat_X.shape[0],mat_X.shape[1]))
    mat_crit = np.zeros((len(X0),Npt))
    vec_crit_init = np.zeros(len(X0)) #critere init (ie compos=Xinit[i], le reste = val vraie)
    vec_crit_end = np.zeros(len(X0)) #critere end (ie compos=Xend[i], le reste = val vraie)
    #mat_crit_init : i colonne = X0 sauf la i ligne qui est Xinit[i]
    mat_crit_init = mat_X0.copy()
    mat_crit_end = mat_X0.copy()
    mat_crit_init[range(len(X0)),range(len(X0))] = Xinit
    mat_crit_end[range(len(X0)),range(len(X0))] = Xend
    for i in range(len(X0)):
        list_mat_X[i,:,:] = mat_X0
        list_mat_X[i,i,:] = mat_X[i,:]
    
    for i in range(len(X0)):
        vec_crit_init[i] = e.mlogV_zg_known(mat_crit_init[:,i],*arg)
        vec_crit_end[i] = e.mlogV_zg_known(mat_crit_end[:,i],*arg)
        for j in range(Npt):            
            X = list_mat_X[i,:,j]        
            #if i == 0 : print str(i),['{:3.3f}'.format(X[i]) for i in range(len(X))]          
            mat_crit[i,j] = e.mlogV_zg_known(X,*arg)
    
    #Pour éviter que les echelles des plots soient biaisées
    #on ne trace pas les 1e14 (remplacement par nan)
    idx = np.where(mat_crit>=1e14)
    mat_crit[idx] = np.nan
    #Subplot 3x3 -> Tvol
    f_Tvol = plt.figure()
    for i in range(9):
        f_Tvol.add_subplot(3,3,i+1)
        plt.plot(mat_X[i,:],mat_crit[i,:])
        if plt_init : plt.plot(Xinit[i],vec_crit_init[i],'or')
        if plt_end : plt.plot(Xend[i],vec_crit_end[i],'og')
        plt.axvline(X0[i],ymin=0,ymax=0.5,color='red',ls='--')        
        if i<3: 
            #plt.ylim((3.5e3,4e3))
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')
            ymin = np.nanmin(mat_crit[i,:])-0.05*np.nanmin(mat_crit[i,:])
            ymax = np.nanmin(mat_crit[i,:])+0.05*np.nanmin(mat_crit[i,:])
            plt.ylim((ymin,ymax))
            plt.grid()
        else:
            #pdb.set_trace()
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')                      
            #plt.ylim((ymin,ymax))
            plt.grid()
        plt.xlabel(list_name[i])        
        
    #Subplot 3x3 -> Tgro
    f_Tground = plt.figure()
    for i in range(9):
        f_Tground.add_subplot(3,3,i+1)
        plt.plot(mat_X[i+9,:],mat_crit[i+9,:])
        plt.axvline(X0[i+9],ymin=0,ymax=0.5,color='red',ls='--')
        if plt_init : plt.plot(Xinit[i+9],vec_crit_init[i+9],'or')
        if plt_end : plt.plot(Xend[i+9],vec_crit_end[i+9],'og')
        if i<3 : 
            #plt.ylim((3.5e3,4e3))
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')
            ymin = np.nanmin(mat_crit[i,:])-0.05*np.nanmin(mat_crit[i,:])
            ymax = np.nanmin(mat_crit[i,:])+0.05*np.nanmin(mat_crit[i,:])
            plt.ylim((ymin,ymax))
            plt.grid()
        else:
            #pdb.set_trace()
            plt.yscale('linear')                     
            plt.ylabel('mlogV(linear)')            
            #ymin = 3539.8
            #ymax = 3540.6
            #plt.ylim((ymin,ymax))
            plt.grid()
        plt.xlabel(list_name[i+9])        
        
    #Subplot 3x3 -> (hv,sigv,gt12,gt13,gt23)
    f_last = plt.figure()
    for i in range(5):
        f_last.add_subplot(2,3,i+1)
        plt.plot(mat_X[i+18,:],mat_crit[i+18,:])
        if plt_init : plt.plot(Xinit[i+18],vec_crit_init[i+18],'or')
        if plt_end : plt.plot(Xend[i+18],vec_crit_end[i+18],'og')
        if i<2 : 
            plt.yscale('linear')    
        else:
            plt.yscale('linear')  
            #plt.ylim((3.54e3,3.6e3))
        plt.grid(True)
        plt.ylabel('mlogV(linear)')            
        plt.axvline(X0[i+18],ymin=0,ymax=0.5,color='red',ls='--')
        plt.xlabel(list_name[i+18])        
        
def plot_mlogV_zg_connu_2D(param):
    
    arg=(param.get_upsilon_gt(),
         param.get_kzlist(),
         np.cos(param.theta),
         param.Na,param.N,
         param.z_g)                          
    
    """ Plot 2D du critère en fonction de hv et sigv"""
    X0 = np.concatenate((mb.get_vecTm_from_Tm(param.T_vol),
                         mb.get_vecTm_from_Tm(param.T_ground),
                         np.array([param.h_v]),
                         np.array([param.sigma_v]),
                         param.get_gtlist()))
                             
    Nhv = 450
    Nsigv = 450
    vec_bcr = np.diag(npl.inv(param.get_fisher_zg_known()))
    dhv = np.sqrt(vec_bcr[18])
    dsigv = np.sqrt(vec_bcr[19])
    
    vec_hv = np.linspace(param.h_v-dhv,param.h_v+dhv,Nhv)
    #vec_sigv = np.linspace(param.sigma_v-dsigv,param.sigma_v+dsigv,Nsigv)
    vec_sigv = np.linspace(0,param.sigma_v+dsigv,Nsigv)
    mat_crit = np.zeros((Nhv,Nsigv))
    
    for ihv,hv in enumerate(vec_hv):
        for isigv,sigv in enumerate(vec_sigv):
            X = X0.copy()
            X[18] = hv
            X[19] = sigv
            mat_crit[ihv,isigv] = e.mlogV_zg_known(X,*arg)
               
    plt.figure()
    bl.iimshow(mat_crit,vec_lin=vec_hv,vec_col=vec_sigv,
               plot_contour='',show_min='',
               zlim=(3e3,1e4),zscale='log')
         
    
if __name__ == '__main__':
    
    np.set_printoptions(precision=3,linewidth=500)
    param = lp.load_param('DB_3')
    param.h_v=30
    param.gammat = np.ones((3,3))*0.95

    #plot_analyse_mlogV_zg_connu_noapriori(param)
    histogramme_autour_de_la_valeur_vraie(param)