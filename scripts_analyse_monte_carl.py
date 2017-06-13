
# -*- coding: utf-8 -*-
""" Scripts d'analyse monte_carlo"""

import os
import basic_lib as bl
import estimation as e 
import numpy as np
import load_param as lp
import plot_tomo as pt
import matplotlib.pyplot as plt
import RVoG_MB as mb
import pdb
import numpy.linalg as npl
import platform as ptm
import sys

def mc_J_J1_V(param,dict_simu,dirname):
    """Script pour effectuer une analyse montecarlo
    des critères de vraissemb et J1 et J2 (cas scalaire)
    
    **Entrées**
        * *param* : jeu de paramètres RVoG
        * *dict_simu* : dictionnaire contenant les paramètres de simulation
        
            * *dict_simu['NbN']* : nombre de taille d'échantillon
            * *dict_simu['Nmin']* : valeurs min de N
            * *dict_simu['Nmax']* : valeurs max de N
            * *dict_simu['P']* : nombre de realisations
            * *dict_simu['UU0']* : initial guess pour mlogV_zg_known 
            * *dict_simu['U0']* : initial guess pour J_scal
            
        * *dirname* : emplacement de sauvegarde
    """
    
    #Calcul
    mat_hvJ,mat_sigvJ,mat_gtJ,mat_bJ,\
    mat_hvJ2,mat_sigvJ2,mat_gtJ2,mat_bJ2,\
    mat_hvV,mat_sigvV,mat_gtV,\
    mean_hvJ,mean_sigJ,mean_gtJ,mean_bJ,\
    mean_hvJ2,mean_sigJ2,mean_gtJ2,mean_bJ2,\
    mean_hvV,mean_sigV,mean_gtV,\
    var_hvJ,var_sigJ,var_gtJ,var_bJ,\
    var_hvJ2,var_sigJ2,var_gtJ2,var_bJ2,\
    var_hvV,var_sigV,var_gtV,\
    vec_N,P,mat_reu = e.monte_carl_estim_mlogV_zg_known_et_J_scal(param,dict_simu)

    #pdb.set_trace()
    #Calcul bcrhv associée
    vec_bcr_hv=np.zeros(vec_N.size)
    for idxN,N in enumerate(vec_N):
        param.N=N
        vec_bcr_hv[idxN] = mb.bcr_hv_mb_zg_connu(param)
    
    if not os.path.exists(dirname):        
        os.makedirs(dirname)        
    #Save
    bl.save_txt_param(param,dirname,)
    bl.save_txt_simu(dict_simu,dirname)
    #resultats    
    #J
    np.save(dirname+'//'+'mat_hvJ',mat_hvJ)
    np.save(dirname+'//'+'mat_sigvJ'+'.npy',mat_sigvJ)
    np.save(dirname+'//'+'mat_gtJ'+'.npy',mat_gtJ)
    np.save(dirname+'//'+'mat_bJ'+'.npy',mat_bJ)
    #J2
    np.save(dirname+'//'+'mat_hvJ2'+'.npy',mat_hvJ2)
    np.save(dirname+'//'+'mat_sigvJ2'+'.npy',mat_sigvJ2)
    np.save(dirname+'//'+'mat_gtJ2'+'.npy',mat_gtJ2)
    np.save(dirname+'//'+'mat_bJ2'+'.npy',mat_bJ2)
    #V
    np.save(dirname+'//'+'mat_hvV'+'.npy',mat_hvV)
    np.save(dirname+'//'+'mat_sigvV'+'.npy',mat_sigvV)
    np.save(dirname+'//'+'mat_gtV'+'.npy',mat_gtV)
    
    #moyennes    
    #J
    np.save(dirname+'//'+'mean_hvJ'+'.npy',mean_hvJ)
    np.save(dirname+'//'+'mean_sigJ'+'.npy',mean_sigJ)
    np.save(dirname+'//'+'mean_gtJ'+'.npy',mean_gtJ)
    np.save(dirname+'//'+'mean_bJ'+'.npy',mean_bJ)
    #J2    
    np.save(dirname+'//'+'mean_hvJ2'+'.npy',mean_hvJ2)
    np.save(dirname+'//'+'mean_sigJ2'+'.npy',mean_sigJ2)
    np.save(dirname+'//'+'mean_gtJ2'+'.npy',mean_gtJ2)
    np.save(dirname+'//'+'mean_bJ2'+'.npy',mean_bJ2)
    #V
    np.save(dirname+'//'+'mean_hvV'+'.npy',mean_hvV)
    np.save(dirname+'//'+'mean_sigV'+'.npy',mean_sigV)
    np.save(dirname+'//'+'mean_gtV'+'.npy',mean_gtV)
    
    #Variances
    #J
    np.save(dirname+'//'+'var_hvJ'+'.npy',var_hvJ)
    np.save(dirname+'//'+'var_sigJ'+'.npy',var_sigJ)
    np.save(dirname+'//'+'var_gtJ'+'.npy',var_gtJ)
    np.save(dirname+'//'+'var_bJ'+'.npy',var_bJ)
    #J2    
    np.save(dirname+'//'+'var_hvJ2'+'.npy',var_hvJ2)
    np.save(dirname+'//'+'var_sigJ2'+'.npy',var_sigJ2)
    np.save(dirname+'//'+'var_gtJ2'+'.npy',var_gtJ2)
    np.save(dirname+'//'+'var_bJ2'+'.npy',var_bJ2)
    #V
    np.save(dirname+'//'+'var_hvV'+'.npy',var_hvV)
    np.save(dirname+'//'+'var_sigV'+'.npy',var_sigV)
    np.save(dirname+'//'+'var_gtV'+'.npy',var_gtV)    
    #vec_N et P
    np.save(dirname+'//'+'vec_N'+'.npy',vec_N)
    np.save(dirname+'//'+'P'+'.npy',P)
    #bcr
    np.save(dirname+'//'+'vec_bcr_hv'+'.npy',vec_bcr_hv)
    #reusiite
    np.save(dirname+'//'+'mat_reu'+'.npy',mat_reu)
    
def mc_V_Tv_Tg_known(param,dict_simu,dirname):
    """Script pour effectuer une analyse montecarlo
    de critère de vraissemb a Tv,Tg,zg connu
    
    **Entrées**
        * *param* : jeu de paramètres RVoG
        * *dict_simu* : dictionnaire contenant les paramètres de simulation
        
            * *dict_simu['NbN']* : nombre de taille d'échantillon
            * *dict_simu['Nmin']* : valeurs min de N
            * *dict_simu['Nmax']* : valeurs max de N
            * *dict_simu['P']* : nombre de realisations
            * *dict_simu['UU0']* : initial guess pour mlogV_zg_known 
            * *dict_simu['U0']* : initial guess pour J_scal
            
        * *dirname* : emplacement de sauvegarde"""
    
    #Calcul
    mat_hvV,mat_sigvV,mat_gtV,\
    mean_hvV,mean_sigV,mean_gtV,\
    var_hvV,var_sigV,var_gtV,\
    vec_N,P= e.monte_carl_estim_mlogV_Tv_Tg_zg_known(param,dict_simu)

    #pdb.set_trace()
    #Calcul bcrhv associée
    vec_bcr_hv=np.zeros(vec_N.size)
    for idxN,N in enumerate(vec_N):
        param.N=N
        vec_bcr_hv[idxN] = mb.bcr_hv_mb_Tv_Tg_zg_connu(param)
    
    if not os.path.exists(dirname):        
        os.makedirs(dirname)        
    #Save
    bl.save_txt_param(param,dirname,)
    bl.save_txt_simu(dict_simu,dirname)
    #resultats    
    #V
    np.save(dirname+'//'+'mat_hvV'+'.npy',mat_hvV)
    np.save(dirname+'//'+'mat_sigvV'+'.npy',mat_sigvV)
    np.save(dirname+'//'+'mat_gtV'+'.npy',mat_gtV)
    
    #moyennes    
    #V
    np.save(dirname+'//'+'mean_hvV'+'.npy',mean_hvV)
    np.save(dirname+'//'+'mean_sigV'+'.npy',mean_sigV)
    np.save(dirname+'//'+'mean_gtV'+'.npy',mean_gtV)
    
    #Variances
    #V
    np.save(dirname+'//'+'var_hvV'+'.npy',var_hvV)
    np.save(dirname+'//'+'var_sigV'+'.npy',var_sigV)
    np.save(dirname+'//'+'var_gtV'+'.npy',var_gtV)    
    #vec_N et P
    np.save(dirname+'//'+'vec_N'+'.npy',vec_N)
    np.save(dirname+'//'+'P'+'.npy',P)
    #bcr
    np.save(dirname+'//'+'vec_bcr_hv'+'.npy',vec_bcr_hv)



def showcurv(dirname,param):
    
    ddata = bl.load_npy_data(dirname)
    #histogramme
    plt.figure();plt.hist(ddata['mat_hvV'][:,0]);plt.title('hvV')
    #stdhvJ et stdhvV=f(N)
    vec_N=ddata['vec_N']
    #bcr hv    
    #vec_bcr_hv=ddata['vec_bcr_hv']
    
    vec_bcr_hv=np.zeros(np.size(vec_N))
    for idxN,N in enumerate(vec_N):
        param.N=N
        vec_bcr_hv[idxN] = mb.bcr_hv_mb_zg_connu(param)

    std_hvJ2=np.sqrt(ddata['var_hvJ2'])
    std_hvV = np.sqrt(ddata['var_hvV'])
    P = ddata['P']
    
    int_conf_stdJ2 = std_hvJ2*1/np.sqrt(2*P-1)
    int_conf_stdV = std_hvV*1/np.sqrt(2*P-1)
    Err=np.vstack((int_conf_stdJ2,int_conf_stdV))
    Err=Err.T
    Y=np.vstack((std_hvJ2,std_hvV))
    Y=Y.T
    pt.plot_err(vec_N,std_hvV,int_conf_stdV,Y_vrai=np.sqrt(vec_bcr_hv),xscale='log',yscale='log')
    plt.title('std_hvV')

def show_var_curve(dirname,param):
    """Afficher les courbes std vs bcr sotckées dans dirname"""
    ddata = bl.load_npy_data(dirname)
    vec_N=ddata['vec_N']
    vec_bcr_hv=ddata['vec_bcr_hv']
    P = ddata['P']
    std_hvV = np.sqrt(ddata['var_hvV'])
    int_conf_stdV = std_hvV*1/np.sqrt(2*P-1)
    pt.plot_err(vec_N,std_hvV,int_conf_stdV,Y_vrai=np.sqrt(vec_bcr_hv),xscale='log',yscale='log')
    
def show_biais_curve(dirname,param):
    """Afficher les courbes std vs bcr sotckées dans dirname"""
    ddata = bl.load_npy_data(dirname)
    vec_N=ddata['vec_N']    
    P = ddata['P']
    std_hvV = np.sqrt(ddata['var_hvV'])
    mean_hvV = ddata['mean_hvV']
    int_conf_meanV = std_hvV*1/np.sqrt(P)    
    pt.plot_err(vec_N,mean_hvV-param.h_v,int_conf_meanV,Y_vrai=np.zeros(vec_N.size),xscale='log',yscale='linear')
    
    
    