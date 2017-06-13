# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:46:59 2015

@author: capdessus
"""
from __future__ import division
from copy import copy

import tomosar_synth as tom
import plot_tomo as pt
import RVoG_MB as mb
import basic_lib as bl
import estimation as e
import load_param as lp
import numpy as np
import numpy.linalg as npl
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as opti

import pdb
plt.ion()

def test_equivalence_normalisation():
    
    #Verifie que faire normlexico+Ti=T+Tebaldini pareil que 
    #Ti=T+Tebaldini
    A = 0.95
    E = 71
    Na = 2
    Np =3
    k_z = [0.1,0.2]
    param_SB = mb.param_rvog(Na)
    param_SB.N = 10000
    param_SB = mb.rvog_reduction(param_SB,A,E)
    param_SB.k_z = [k_z[0]]
    param_SB.z_g = 0
    W_k_vrai_SB = tom.UPS_to_MPMB(param_SB.get_upsilon(),2)
    data_synth = tom.TomoSARDataSet_synth(Na,param_SB)    
    taille_test = param_SB.N
    W_k1= data_synth.get_W_k_norm_rect(taille_test,Na,'mat+ps+tebald')
    W_k2= data_synth.get_W_k_norm_rect(taille_test,Na,'ps+tebald')
    np.set_printoptions(linewidth=150,precision=2)
    
    R_t1,C_t1 = tom.sm_separation(W_k1,Np,Na)             
    interv_a1,interv_b1,Cond1,alpha1 = tom.search_space_definition(R_t1,C_t1,Na) 

    R_t2,C_t2 = tom.sm_separation(W_k2,Np,Na)             
    interv_a2,interv_b2,Cond2,alpha2 = tom.search_space_definition(R_t2,C_t2,Na) 
                    
    print '1'
    print 'a',interv_a1
    print 'b',interv_b1
    print 'gamma_1_1',R_t1[0][0,1]
    print 'gamma_2_1',R_t1[1][0,1]
    
    print '\n\n2'
    print 'a',interv_a2
    print 'b',interv_b2
    print 'gamma_1_2',R_t2[0][0,1]
    print 'gamma_2_2',R_t2[1][0,1]


def test_plot_cu_plus_tebaldini_plus_legend():
    Na = 3
    Np = 3
    A = 0.75
    E = 71
    k_z = [0.1,0.2]
    ant1=0
    ant2=1
    #MB
    param_MB = mb.param_rvog()
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z = k_z
    param_MB.z_g = 0
    param_MB.theta=np.pi/4
    param_MB.Na=len(param_MB.k_z)+1
    param_MB.h_v=30
    param_MB.sigma_v=0.0345
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon(),Na)

    data_tomo=tom.TomoSARDataSet_synth(param_MB)    
    W_k_noise=data_tomo.get_W_k_rect(param_MB,nb_echant=1000)
    
    W_k_MB = W_k_noise
    
    R_t_MB,C_t_MB,G_MB = tom.sm_separation(W_k_MB,Np,Na)             
    interv_a_MB,interv_b_MB,Cond,alpha = tom.search_space_definition(R_t_MB,C_t_MB,Na) 
    
    vec_a_MB = interv_a_MB[0][1]
    vec_b_MB = np.linspace(interv_b_MB[0][0],interv_b_MB[0][1],50)
    dist_g_MB,dist_v_MB = tom.proximity_C_T(W_k_MB,Na,R_t_MB,C_t_MB,\
                                vec_a_MB,vec_b_MB,param_MB)
    
        
    _,R_g_min_MB,R_v_min_MB,_,_ = tom.value_R_C(R_t_MB,C_t_MB,interv_a_MB[0][0],\
                                    vec_b_MB[np.argmin(dist_g_MB)])
    
    Ups = tom.UPS_to_MPMB(W_k_MB,Na)
    pt.plot_cu_plus_tebaldini_plus_legend(Ups,R_t_MB,R_g_min_MB,\
                                       R_v_min_MB,\
                                       interv_a_MB,interv_b_MB,\
                                       ant1,ant2)
                                           
def test_evol_domaine_possib_SB_vs_MB(data_type='pure'):
    print 'test_evol_domaine_possib_SB_vs_MB'
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.2]    
    #SB
    param_SB = mb.param_rvog(2)
    param_SB = mb.rvog_reduction(param_SB,A,E)
    param_SB.k_z = [k_z[0]]
    W_k_vrai_SB = tom.UPS_to_MPMB(param_SB.get_upsilon(),2)
    #MB
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z = k_z
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon(),Na)    
    
    
    if data_type=='pure':
        W_k_SB = tom.UPS_to_MPMB(param_SB.get_upsilon(),2)
        W_k_MB = tom.UPS_to_MPMB(param_MB.get_upsilon(),Na)
    elif data_type =='noisy':       
        data_MB = tom.TomoSARDataSet_synth(Na,param_MB)    
        data_SB = tom.TomoSARDataSet_synth(2,param_SB)            
        taille_test = 100      
        W_k_MB = data_MB.get_W_k_norm_rect(taille_test,Na)
        W_k_SB = data_SB .get_W_k_norm_rect(taille_test,2)

    #sm_separation SB
    W_k_SB_norm = tom.normalize_MPMB_PS_Tebald(W_k_SB,2)
    R_t_SB,C_t_SB,G_SB = tom.sm_separation(W_k_SB_norm,Np,2)             
    interv_a_SB,interv_b_SB,Cond_SB,alpha_SB =\
        tom.search_space_definition(R_t_SB,C_t_SB,2)  
    
    SKP_1 = np.kron(C_t_SB[0],R_t_SB[0])+np.kron(C_t_SB[1],R_t_SB[1])
    diff_1 = G_SB - SKP_1
    norm_SB = np.trace(diff_1.dot(diff_1.T.conj()))
    print '||G_SB-SKP_1||',norm_SB
    
    
    #print param_MB.k_z
    #print param_MB.get_k_z(0,1,Na)
        
    #sm_separation MB
    W_k_MB_norm = tom.normalize_MPMB_PS_Tebald(W_k_MB,Na)
    R_t_MB,C_t_MB,G_MB = tom.sm_separation(W_k_MB_norm,Np,Na)             
    interv_a_MB,interv_b_MB,Cond,alpha = tom.search_space_definition(R_t_MB,C_t_MB,Na) 
    
    SKP_2 = np.kron(C_t_MB[0],R_t_MB[0])+np.kron(C_t_MB[1],R_t_MB[1])
    diff_2 = G_MB - SKP_2
    norm_MB = np.trace(diff_2.dot(diff_2.T.conj()))
    print '||G_MB-SKP_2||',norm_MB
    
    
    vec_a_SB = interv_a_SB[0][0]
    vec_b_SB = np.linspace(interv_b_SB[0][0],interv_b_SB[0][1],50)
    vec_a_MB = interv_a_MB[0][0]
    vec_b_MB = np.linspace(interv_b_MB[0][0],interv_b_MB[0][1],50)
    

    #SB
    dist_g_SB,dist_v_SB = tom.proximity_C_T(W_k_SB,2,R_t_SB,C_t_SB,\
                                        vec_a_SB,vec_b_SB,param_SB)
    
    _,R_g_min_SB,R_v_min_SB,C_g_min_SB,C_v_min_SB =\
                                tom.value_R_C(R_t_SB,C_t_SB,interv_a_SB[0][0],\
                                    vec_b_SB[np.argmin(dist_g_SB)])
                                    
    T_g,T_v = tom.denormalisation_teb(W_k_SB,2,C_g_min_SB,C_v_min_SB)
    
    
    SKP_1min = np.kron(C_v_min_SB,R_v_min_SB)+\
                            np.kron(C_g_min_SB,R_g_min_SB)
    diff_min1 = G_SB - SKP_1min
    norm_SB_min = np.trace(diff_min1.dot(diff_min1.T.conj()))
    np.set_printoptions(precision=5)
    
    print '/////////////// TEST MB vs SV /////////////////////'
    print '======================= SB ==================='
    print '-- Polar --'
    print '||G_SB-SKP_min1||',norm_SB_min    
    print 'ground'
    print T_g
    print param_SB.T_ground*param_SB.get_a()    
    print '\nvol'
    print T_v
    print param_SB.T_vol*param_SB.get_I1()
    print '-- Structure --'
    print 'R_g_min_SB'
    print R_g_min_SB
    print 'R_v_min_SB'
    print R_v_min_SB
    print 'R_v_vrai'
    print param_SB.get_R_v()        
    #MB
    dist_g_MB,dist_v_MB = tom.proximity_C_T(W_k_MB,Na,R_t_MB,C_t_MB,\
                                vec_a_MB,vec_b_MB,param_MB)
    
    _,R_g_min_MB,R_v_min_MB,C_g_min_MB,C_v_min_MB =\
                                tom.value_R_C(R_t_MB,C_t_MB,interv_a_MB[0][0],\
                                    vec_b_MB[np.argmin(dist_g_MB)])
    T_g,T_v = tom.denormalisation_teb(W_k_MB,Na,C_g_min_MB,C_v_min_MB)        
    SKP_2min = np.kron(C_v_min_MB,R_v_min_MB)+\
                            np.kron(C_g_min_MB,R_g_min_MB)
    diff_min2 = G_MB- SKP_2min
    norm_MB_min = np.trace(diff_min2.dot(diff_min2.T.conj()))
    
    plt.figure()
    plt.plot(vec_b_MB,dist_g_MB[0,:],'-*')
    print '===================== MB ===================='
    print '||G_MB_MB-SKP_min2||',norm_MB_min
    print 'ground'
    print T_g
    print param_MB.T_ground*param_MB.get_a()    
    print '\nvol'
    print T_v
    print param_MB.T_vol*param_MB.get_I1()
    print '-- Structure --'
    print 'R_g_min_MB'
    print R_g_min_MB
    print 'R_v_min_MB'
    print R_v_min_MB
    print 'R_v_vrai'
    print param_MB.get_R_v()

def test_denormalisation_tebaldini():
    """
    A=np.array([[1,2],[3,4]])
    #B=np.array([[5,6,7],[8,9,10],[11,12,13]])
    B=np.array([[5,6],[8,9]])
    A_kro_B=np.kron(A,B)
    P_t=np.transpose(tom4.p_rearg(A_kro_B,Np,Na))
    BxA = tom4.inv_p_rearg(P_t,Np,Na)
    
    B_kro_A = np.kron(B,A)
    """

    T_vol =np.random.random_integers(0,9,(3,3))
    T_sol =np.random.random_integers(0,9,(3,3))
    R_sol =np.random.random_integers(0,9,(3,3))
    R_vol =np.random.random_integers(0,9,(3,3))
    Na=3
    Np=3
    Ups = np.kron(R_sol,T_sol)+np.kron(R_vol,T_vol) 
    W = tom.UPS_to_MPMB(Ups)
    E = np.diag(np.diag(W.copy())) #Matrice des coeffs diagonaux de E
    W_norm = tom.normalize_MPMB2(W,Na)
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)  
    F= tom.power(np.diag(np.array([E[0,0],E[2,2], E[4,4]])),0.5)
    print F.dot(C_t[1].dot(F))
    print T_vol
    
def test_commutte_ejd():
    A_t = np.random.random_integers(0,9,(3,3))
    A_t = 1/2*(A_t+A_t.T.conj()) #hermitianisation de A_t
    B_t = np.random.random_integers(0,9,(3,3))
    B_t = 1/2*(B_t+B_t.T.conj()) #hermitianisation de B_t
    
    D1,D2,A,LAMBDA_mat = tom.ejd2(A_t,B_t)
    D1_p,D2_p,A_p,LAMBDA_mat_p = tom.ejd2(B_t,A_t)        
    print LAMBDA_mat
    print LAMBDA_mat_p
    return A_t,B_t,D1,D2,A,LAMBDA_mat,D1_p,D2_p,A_p,LAMBDA_mat_p

def test_ecart_angle_droite():
    
    plt.close('all')
    Np=3
    Na=2
    param = mb.param_rvog(Na)
    A= [0.1,0.95]
    E= [71,200]
    k_z = [0.1]
    param.k_z = [k_z[0]]        
    param.display()
    #calcul pour plusieurs A et E pour differentes N (taille echant)    
    min_taille1 = 50
    max_taille1 = 10000
    nb_taille1 = 20
    varia_taille_echant1 = np.floor(np.logspace(np.log10(min_taille1),\
                           np.log10(max_taille1),nb_taille1))                           
    varia_taille_echant1 = varia_taille_echant1.astype(np.int64)
    N_real = 5000
    
    plot_hist0=0;save_hist0=0;plot_cohe0=0;save_data0=1
    tom.ecart_angle_inclin_A_E(A,E,param,varia_taille_echant1,N_real,\
                                save_data0,plot_hist0,save_hist0,plot_cohe0)
    sauve_plot_biais_variance = 1
    pt.plot_bias_var_inclin_A_E(A,E,varia_taille_echant1,N_real,sauve_plot_biais_variance)
    
def test_angle_ff():
    """Calcul de differetens angle d'inclinaison 
    de matrice Upsilon non bruités et plot """
    Np=3
    Na=2
    A = 0.1
    E = 200
    param = mb.param_rvog(Na)
    param = mb.rvog_reduction(param,A,E)
    Nb_k_z = 5
    vec_k_z = np.linspace(0.1,2*np.pi/param.h_v,Nb_k_z)
    print vec_k_z
    theta = np.zeros((Nb_k_z,1))
    for i,k_z in enumerate(vec_k_z):
        param.k_z = [k_z]
        Ups = param.get_upsilon()
        omega = tom.polinsar_compute_omega12blanchi(Ups)
        theta[i],_ = tom.polinsar_estime_droite(omega)     
        plt.hold(True)
        tom.polinsar_plot_cu(Ups)
    print theta*180/np.pi
    
def load_ups_theta():
    A_test = [0.1,0.95]
    E_test = [71,200]
    for A in A_test:
        for E in E_test:                        
            date = '8_10/'
            home_dir ='/home/capdessus/Python/Code_Pierre/'
            folder_name = 'data/angle_inclinaison_droite/'+date
            sub_folder_name ='A_{0}_E_{1}/'.format(A,E)
            total_path = home_dir+folder_name+sub_folder_name                
            varia_taille_echant,theta_vrai,\
            theta_pascale_moy,theta_pascale_var,\
            theta_tebald_moy,theta_tebald_var,\
            Ups= pt.load_data(total_path)            
            print 'A= {0} E ={1}\t theta_vrai {2}'.format(A,E,theta_vrai*180/np.pi)
            print 'Ups'
            bl.printm(Ups)

def test_matrice_antoine():
    Na=2
    Np=3
    path= '/home/capdessus/Python/Code_Pierre/matrice_Ups_test_.txt'
    path2= '/home/capdessus/Python/Code_Pierre/matrice_Ups_test_norm.txt'
    Ups_antoine=bl.load_txt(path)
    Ups_norm_antoine=bl.load_txt(path2)
    A=0.1 
    E=71    
    param = mb.param_rvog(Na)
    param = mb.rvog_reduction(param,A,E)
    Ups_vrai = param.get_upsilon()
    omega_vrai_blanc = tom.polinsar_compute_omega12blanchi(Ups_vrai)
    theta_vrai,_ = tom.polinsar_estime_droite(omega_vrai_blanc)
 
    omega_antoine = tom.polinsar_compute_omega12blanchi(Ups_antoine)
    theta_antoine,_ = tom.polinsar_estime_droite(omega_antoine)     
        
    W_k=tom.UPS_to_MPMB(Ups_antoine,Na)
    W_k_norm=tom.normalize_MPMB_PS_Tebald(W_k,Na)
    
    #sm_separation  
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,Na)                 
    interv_a,interv_b,Cond,alpha =\
                   tom.search_space_definition(R_t,C_t,Na)          
    a=0; b=1
    _,R1,R2,C1,C2=tom.value_R_C(R_t,C_t,a,b)        
    gamma1 = R1[0,1]; gamma2 = R2[0,1]
    gamma = [gamma1,gamma2]
    theta_tebald_antoine= bl.estime_line_svd(gamma,theta_vrai)
    print '-------------------------------'
    print 'Ups_antoine'
    bl.printm(Ups_antoine)
    print 'Ups_antoine_norm'
    bl.printm(Ups_norm_antoine)    
    print 'Rtilde1'    
    bl.printm(R_t[0])
    print 'Rtilde2'
    bl.printm(R_t[1])
    print 'Ctilde1'    
    bl.printm(C_t[0])
    print 'Ctilde2'
    bl.printm(C_t[1])
    print 'R1'    
    bl.printm(R1)
    print 'R2'
    bl.printm(R2)
    print 'C1'
    bl.printm(C1)
    print 'C2'
    bl.printm(C2)
    
    print 'theta_vrai {0}'.format(theta_vrai*180/np.pi)    
    print 'theta_antoine_FF {0}'.format(theta_antoine*180/np.pi)
    print 'theta_antoine_Tebald {0}'.format(theta_tebald_antoine*180/np.pi)
    
    tom.polinsar_plot_cu(tom.MPMB_to_UPS(W_k_norm,Na),title =' CU')
    #tom.polinsar_plot_cu(Ups_norm_antoine,title =' CU')
    
    print '------ Influcence normalisation ------------'
    print 'Ups_antoine'
    bl.printm(Ups_antoine)
    print 'Ups_antoine_norm'
    bl.printm(Ups_norm_antoine)    
    print 'Ups normalisé par Pierre à partir de Ups_antoine'
    bl.printm(tom.MPMB_to_UPS(W_k_norm,Na))
    return Ups_antoine,Ups_norm_antoine,tom.MPMB_to_UPS(W_k_norm,Na)
    
    
def test_estimaeur_dpctm_kz():

    hv,sig,gt1,gt2,gt3,\
    meanhv,varhv,meansig,varsig,\
    meangt1,vargt1,meangt2,vargt2,\
    meangt3,vargt3 = monte_carl_estim_dpct_kz(param_MB)
    
    np.save('hv',hv)
    np.save('sig',sig)
    np.save('gt1',gt1)
    np.save('gt2',gt2)
    np.save('gt3',gt3)

    Nb_N=15#nb de taille diff
    vec_N = np.floor((np.logspace(2,5,Nb_N)))
    P_real=100
    
    pt.plot_biais_variance_err(vec_N,meanhv,varhv,P_real,\
                            'hv_moy','hv_var',param_MB.h_v)
    pt.plot_biais_variance_err(vec_N,meansig,varsig,P_real,\
                            'sig_moy','sig_var',param_MB.sigma_v)
    pt.plot_biais_variance_err(vec_N,meangt1,vargt1,P_real,\
                            'gt1_moy','gt1_var',param_MB.gammat[0,1])                            
    pt.plot_biais_variance_err(vec_N,meangt2,vargt2,P_real,\
                            'gt2_moy','gt2_var',param_MB.gammat[0,2])                            
    pt.plot_biais_variance_err(vec_N,meangt3,vargt3,P_real,\
                            'gt3_moy','gt3_var',param_MB.gammat[1,2])                            

def test_estim_ecart_ang_pol():

    Na = 3
    Np = 3
    A = 0.95
    E = 4000
    k_z = [0.1,0.15]
    #MB
    
    param_MB = mb.param_rvog()
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.Na=len(param_MB.k_z)+1
    param_MB.theta=45*np.pi/180
    param_MB.sigma_v=0.0345
    param_MB.h_v=30
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    #param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    param_MB.gammat=np.ones((3,3))
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt(),Na)
     
    W_k=W_k_vrai_MB 
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param_MB.Na)
    
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param_MB.Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    #choix du a et b
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]
    b = (interv_b[0][0]+interv_b[0][1])/2
    b_vrai = tom.b_true(R_t,param_MB)
    _,Rg,Rv,Cg,Cv=tom.value_R_C(R_t,C_t,a,b_vrai)    
    
    e.estim_ecart_ang_pol(W_k,param_MB)    


def test_estim_ecart_ang():
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB
    
    param_MB = mb.param_rvog()
    param_MB = mb.rvog_reduction(param_MB,A,E)    
    param_MB.k_z=k_z
    param_MB.Na=len(param_MB.k_z)+1
    param_MB.sigma_v=0.0345
    param_MB.theta=45*np.pi/180
    param_MB.h_v=30
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    param_MB.N=100
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt(),Na)
    
    nb_echant=10**3   
    data_synt=tom.TomoSARDataSet_synth(param_MB.Na,param_MB)
    W_k_bruit=data_synt.get_W_k_rect(param_MB,int(nb_echant),param_MB.Na) 
    
    W_k=W_k_vrai_MB
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param_MB.Na)
    R_t,C_t,_ = tom.sm_separation(W_k_vrai_MB,Np,param_MB.Na)
    
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    #*interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    #choix du a et b
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]
    b = (interv_b[0][0]+interv_b[0][1])/2
    b_vrai = tom.b_true(R_t,param_MB)
    _,Rg,Rv,Cg,Cv=tom.value_R_C(R_t,C_t,a,b_vrai)    
    vec_gm = tom.gamma_from_Rv(Rv)
    vec_gt = np.array([param_MB.gammat[0,1],param_MB.gammat[0,2],param_MB.gammat[1,2]])
    vec_kz = param_MB.get_kzlist()
    
    costheta=np.cos(param_MB.theta)
    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
        
    estim=0
    if estim:
        J,J2,Ressemb,ErrQM,\
        vec_hv,vec_sig,vec_b,\
        hv_J,hv_J2,hv_MV,hv_EQM,\
        sig_J,sig_J2,sig_MV,sig_EQM,\
        vec_gt_J,vec_gt_J2,vec_gt_MV,\
        vec_gt_EQM = e.estim_ecart_ang(W_k,param_MB)
     
    #pdb.set_trace()
    load=1
    if(load):
        path = 'D:/PIERRE CAPDESSUS/Python/Code_Pierre/sauv_data_estimation/Test4/'
        J=np.load(path+'J.npy')        
        J2=np.load(path+'J2.npy')        
        #ErrQM = np.load(path+'EQM.npy')        
        Ressemb=np.load(path+'Ressemb.npy')                
        vec_hv=np.load(path+'vec_hv.npy')
        vec_sig=np.load(path+'vec_sig.npy',)
        vec_b=np.load(path+'vec_b.npy',)
        hv_J=np.load(path+'hv_J.npy')
        hv_J2=np.load(path+'hv_J2.npy')
        hv_MV=np.load(path+'hv_MV.npy')
        hv_EQM=np.load(path+'hv_EQM.npy')
        sig_J=np.load(path+'sig_J.npy')
        sig_J2=np.load(path+'sig_J2.npy')
        sig_MV=np.load(path+'sig_MV.npy')
        sig_EQM=np.load(path+'sig_EQM.npy')        
    save =0
    if(save):
        np.save('J',J)        
        np.save('J2',J2)        
        np.save('Ressemb',Ressemb)        
        np.save('ErrQM',EQM)  
        np.save('vec_hv',vec_hv)
        np.save('vec_sig',vec_sig)
        np.save('vec_b',vec_b)
        np.save('hv_J',hv_J)
        np.save('hv_J2',hv_J2)
        np.save('hv_MV',hv_MV)
        np.save('hv_EQM',hv_EQM)
        np.save('sig_J',sig_J)
        np.save('sig_J2',sig_J2)
        np.save('sig_MV',sig_MV)
        np.save('sig_EQM',sig_EQM)
        np.save('vec_gt_J',vec_gt_J)
        np.save('vec_gt_J2',vec_gt_J2)
        np.save('vec_gt_MV',vec_gt_MV)
        np.save('vec_gt_EQM',vec_gt_EQM)

    #calcul de la valeur de critere J sur les hv/sigb selectionnés sur
    #pour chaque b
    
    minJ = np.zeros(vec_b.size,dtype='double')
    minJ2 = np.zeros(vec_b.size,dtype='double')
    minV=np.zeros(vec_b.size)
    minEQM=np.zeros(vec_b.size)
    for idxb,b in enumerate(vec_b):
        idx_hvJ  = np.argmin(np.abs(hv_J[idxb]-vec_hv))
        idx_sigJ  = np.argmin(np.abs(sig_J[idxb]-vec_sig))        
        idx_hvMV  = np.argmin(np.abs(hv_MV[idxb]-vec_hv))
        idx_sigMV  = np.argmin(np.abs(sig_MV[idxb]-vec_sig))        
        #idx_hvEQM  = np.argmin(np.abs(hv_EQM[idxb]-vec_hv))
        #idx_sigEQM  = np.argmin(np.abs(sig_EQM[idxb]-vec_sig))        
                
        minJ[idxb] = J[idxb,idx_hvJ,idx_sigJ]
        minJ2[idxb] = J2[idxb,idx_hvJ,idx_sigJ]
        minV[idxb] = Ressemb[idxb,idx_hvMV,idx_sigMV]
        #minEQM[idxb] = ErrQM[idxb,idx_hvEQM,idx_sigEQM]
    
    b_vrai=tom.b_true(R_t,param_MB)
    idx_bvrai=np.argmin(np.abs(vec_b-b_vrai))
    b_vrai_num=vec_b[idx_bvrai]
    print 'b_vrai: {0}'.format(b_vrai) 
    print 'b_vrai echantillonné: {0} deltab:{1}'.format(b_vrai_num,vec_b[1]-vec_b[0]) 
    print 'b minimisant minJ: {0}'.format(vec_b[np.argmin(minJ)])
    print 'b minimisant minMV: {0}'.format(vec_b[np.argmin(minV)])
    print 'b minimisant minEQM: {0}'.format(vec_b[np.argmin(minEQM)])
    
    
    plt.close('all')
    pt.plot_cu_sm_possible(tom.MPMB_to_UPS(W_k,3),R_t,interv_a,interv_b,\
                        0,1,title='CU')
    plt.hold(True)                        
    plt.polar([np.angle(param_MB.get_gamma_v_gt(0,1)),
               np.angle(param_MB.get_gamma_v(0,1))],
              [np.abs(param_MB.get_gamma_v_gt(0,1)),
               np.abs(param_MB.get_gamma_v(0,1))],'--ko')                        
    plt.draw()
    
    ib=idx_bvrai
    
    idxminJ = zip(*np.where(J[ib,:,:]==np.nanmin(J[ib,:,:])))
    idxminJ2 = zip(*np.where(J2[ib,:,:]==np.nanmin(J2[ib,:,:])))
    idxminMV = zip(*np.where(Ressemb[ib,:,:]==np.nanmin(Ressemb[ib,:,:])))
    #idxminEQM = zip(*np.where(ErrQM[ib,:,:]==np.nanmin(ErrQM[ib,:,:])))
    
    
    idxminminJ = zip(*np.where(minJ==np.nanmin(minJ)))
    idxminminJ2 = zip(*np.where(minJ2==np.nanmin(minJ2)))
    idxminminV = zip(*np.where(minV==np.nanmin(minV)))
    

    print 'b={0}'.format(vec_b[idx_bvrai])
    print 'sigmav_MV={0}'.format(vec_sig[idxminMV[0][0]])
    print 'hv_MV={0}'.format(vec_hv[idxminMV[0][1]])
    print '----------------------------------------'
    print 'sigmav_J={0}'.format(vec_sig[idxminJ[0][0]])
    print 'hv_J={0}'.format(vec_hv[idxminJ[0][1]])
    print '----------------------------------------'
    print 'sigmav_J2={0}'.format(vec_sig[idxminJ2[0][0]])
    print 'hv_J2={0}'.format(vec_hv[idxminJ2[0][1]])
    print 'b={0}'.format(vec_b[idx_bvrai])
    print 'sigmav_MV={0}'.format(sig_MV[np.argmin(minV)])
    print 'hv_MV={0}'.format(hv_MV[np.argmin(minV)])
    print '----------------------------------------'
    print 'sigmav_J={0}'.format(sig_J[np.argmin(minJ)])
    print 'hv_J={0}'.format(hv_J[np.argmin(minJ)])
    print '----------------------------------------'
    print 'sigmav_J2={0}'.format(sig_J2[np.argmin(minJ2)])
    print 'hv_J2={0}'.format(hv_J2[np.argmin(minJ)])    
    plt.figure()
    plt.imshow(J[1,:,:])
    plt.title('ib=1')
    plt.colorbar()
    plt.draw()
    
    plt.figure()
    plt.imshow(J[-1,:,:])
    plt.title('ib=-1')
    plt.colorbar()
    plt.draw()
    plt.figure()
    levels=np.logspace(np.log10(np.min(Ressemb[ib,:,:])),
                       np.log10(np.max(Ressemb[ib,:,:])),25)
    plt.imshow(Ressemb[ib,:,:],origin='lower')
    idx_hv=idxminMV[0][0]    
    idx_sig=idxminMV[0][1]
    plt.plot(idx_sig,idx_hv,'or',label='min MV')    
    plt.colorbar()
    plt.legend()
    plt.axis('tight')
    plt.hold(True)
    plt.contour(Ressemb[ib,:,:],levels,linewidths=2,colors='Black')
    plt.title('Ressemb ib='+str(ib))
    plt.draw()

    """EQM
    plt.figure()
    levelsErrQM=np.logspace(np.log10(np.min(ErrQM[ib,:,:])),
                       np.log10(np.max(ErrQM[ib,:,:])),25)
    plt.imshow(ErrQM[ib,:,:],origin='lower')
    idx_hv=idxminEQM[0][0]    
    idx_sig=idxminEQM[0][1]
    plt.plot(idx_sig,idx_hv,'or',label='min ErrQM')    
    plt.colorbar()
    plt.legend()
    plt.axis('tight')
    plt.hold(True)
    plt.contour(ErrQM[ib,:,:],levelsErrQM,linewidths=2,colors='Black')
    plt.title('ErrQM ib='+str(ib))
    plt.draw()    
    """
    
    font = {'family': 'sans-serif',
            'weight': 'bold',
            'size': 25
            }   
    #plt.rc('font',**font)
    plt.rc('lines',linewidth=2)
    
    plt.rcParams['lines.markersize']=9
    plt.rcParams['axes.labelsize']=25
    
    #plt.rcParams['axes.labelsize']=18
    plt.rcParams['font.size']=20
    
    plt.figure()
    plt.plot(vec_b,hv_J,'b.-',label='J')    
    plt.plot(vec_b,hv_J2,'g.-',label='J2')    
    #plt.plot(vec_b,hv_MV,'r.-',label='MV')    
    #plt.plot(vec_b,hv_EQM,'c.-',label='EQM')    
    #plt.plot(vec_b,param_MB.h_v*np.ones(vec_b.size),'k--',label='hv vrai')
    hv_num = vec_hv[np.argmin(np.abs(vec_hv-param_MB.h_v))]
    plt.plot(vec_b,hv_num*np.ones(vec_b.size),'r--',label='hv vrai')
    plt.plot([b_vrai_num,b_vrai_num],
              [hv_num,hv_num],'ok')
    plt.xlabel('b')
    plt.ylabel('hv')
    plt.grid()
    plt.title('Estimation de hv pour chaque b')
    plt.legend(loc='best')
    plt.draw()
    
    plt.figure()    
    plt.plot(vec_b,sig_J,'b.-',label='J')    
    plt.plot(vec_b,sig_J2,'g.-',label='J2')    
    #plt.plot(vec_b,sig_MV,'r.-',label='MV')    
    #plt.plot(vec_b,sig_EQM,'c.-',label='EQM')    
    sig_num = vec_sig[np.argmin(np.abs(vec_sig-param_MB.sigma_v))]
    plt.plot(vec_b,sig_num*np.ones(vec_b.size),'k--',label='sig vrai (num)')
    plt.plot([b_vrai_num,b_vrai_num],
              [sig_num,sig_num],'ok')    
    plt.xlabel('b')
    plt.ylabel('sigmav')    
    plt.grid()
    plt.legend(loc='best')
    plt.draw()
    
    plt.figure()
    plt.plot(vec_b,minJ,'b.-',label='J')
    plt.plot(vec_b,minJ2,'r.-',label='J2')
    plt.plot(vec_b,(minV),'g.-',label='-logV-med(-logV)')
    plt.axvline(x=b_vrai_num,ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3)
    plt.title('Variation critere en fonction de b')
    plt.xlabel('b')
    plt.grid()
    
    plt.legend(loc='best')
    plt.draw()
    
    plt.figure()
    plt.semilogy(vec_b,minJ-np.min(minJ),'b.-',label='J')
    plt.semilogy(vec_b,minJ2-np.min(minJ2),'r.-',label='J2')
    plt.semilogy(vec_b,(minV)-np.min(minV),'g.-',label='-logV-med(-logV)')
    plt.axvline(x=b_vrai_num,ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3)
    plt.title('Variation critere en fonction de b. (x-min(x))')
    plt.grid()
    plt.xlabel('b')
    plt.legend(loc='best')
    plt.draw()
    
    
    plt.figure()
    plt.plot(vec_b,(minJ-np.min(minJ)),'b.-',label='J')
    plt.plot(vec_b,(minJ2-np.min(minJ2)),'r.-',label='J2')
    plt.plot(vec_b,minV-np.min(minV),'g.-',label='MV')
    #plt.plot(vec_b,(minEQM-np.mi(minEQM))/np.std(minEQM),'c.-',label='EQM')
    plt.axvline(x=b_vrai_num,ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3)
    
    plt.xlabel('b')
    plt.ylabel('')
    plt.title('Variation critere en fonction de b. (x-min(x))')
    plt.grid()
    plt.legend(loc='best')    
    plt.draw()
    

def test_estim_ecart_ang_reestim_b():
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB
    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.sigma_v=0.0345
    param_MB.h_v=30
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    #param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt(),Na)
    W_k_vrai_MB_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k_vrai_MB,param_MB.Na)
    nb_echant=10**5
    
    data_synt=tom.TomoSARDataSet_synth(param_MB.Na,param_MB)
    W_k_bruit=data_synt.get_W_k_rect(param_MB,int(nb_echant),param_MB.Na) 
    
    W_k=W_k_bruit 
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param_MB.Na)
    
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
     
    R_t_vrai,_,_=tom.sm_separation(W_k_vrai_MB_norm,Np,Na)
    
    
    hvJ,hvJ2,hvV,\
    sigJ,sigJ2,sigV,\
    gt_J,gt_J2,gt_V,\
    X_minJ,X_minJ2,X_minV,\
    minJ,minJ2,minV,\
    bminJ,bminJ2,bminV,\
    vec_b,vec_br=e.estim_ecart_ang_reestim_b(W_k,param_MB)

    idx_sort=np.argsort(vec_br)
    vec_brs = np.sort(vec_br)
    minJsort=minJ[idx_sort]
    
    bvrai=tom.b_true2(R_t_vrai,param_MB)
    
    
    ant1,ant2=(0,1)
    gvgtij = param_MB.get_gamma_v_gt(ant1,ant2)
    gvij = param_MB.get_gamma_v(ant1,ant2)
    gvb = bvrai*R_t[0][ant1,ant2]+(1-bvrai)*R_t[1][ant1,ant2]
    gvbJ = bminJ*R_t[0][ant1,ant2]+(1-bminJ)*R_t[1][ant1,ant2]
    
    pt.plot_cu_sm_possible(tom.UPS_to_MPMB(W_k,3),R_t,interv_a,interv_b,ant1,ant2)             
    plt.hold(True)
    plt.polar(np.angle(gvgtij),np.abs(gvgtij),'ok',label='gvgt vrai')
    plt.polar(np.angle(gvij),np.abs(gvij),'sk',label='gv vrai')
    plt.polar(np.angle(gvb),np.abs(gvb),'or',
              label='gv(bvrai) vrai b={0}'.format(bvrai))
    plt.polar(np.angle(gvbJ),np.abs(gvbJ),'oc',
              label='gv(bminJ) b={0}'.format(bminJ))
    plt.legend()
    plt.hold(False)
    
    plt.figure()
    plt.plot(np.linspace(vec_b[0]-0.05,vec_b[-1]+0.05),
             np.polyval(np.polyfit(vec_b,vec_br,1),np.linspace(vec_b[0]-0.05,vec_b[-1]+0.05)),
             label='fit')
    plt.plot(np.linspace(vec_b[0]-0.05,vec_b[-1]+0.05),
             np.linspace(vec_b[0]-0.05,vec_b[-1]+0.05),
             'k--',label='Id')
             
    plt.plot(vec_b,vec_br,'ko-')
    plt.plot(tom.b_true2(R_t,param_MB),tom.b_true2(R_t_vrai,param_MB),'or',label='bvrai')
    plt.legend()
    plt.xlabel('vec_b')
    plt.ylabel('vec_br')
    plt.grid(True)
    
    plt.figure()
    plt.hold(True)
    plt.plot(vec_b,minJ,'*-r',label='b echant')
    plt.plot(vec_brs,minJsort,'sb-',label='b recalc')
    plt.axvline(x=bvrai,
                ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3,label='bvrai={0}'.format(bvrai))
    plt.axvline(x=bminJ,
                ymin=0,ymax=1,alpha=.7,linestyle='--',color='r',
                lw=3,label='bminJ={0}'.format(bminJ))
    
    plt.hold(False)
    plt.grid(True)
    plt.legend()
    plt.xlabel('b')
    plt.ylabel('Critere')
    plt.yscale('log')
    plt.title('Critere =f(b) (echelle log)')
    
    plt.figure()
    plt.hold(True)
    plt.plot(vec_b,minJ,'*-r',label='b echant')
    plt.plot(vec_brs,minJsort,'sb-',label='b recalc')
    plt.axvline(x=bvrai,
                ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3,label='bvrai={0}'.format(bvrai))
    plt.axvline(x=bminJ,
                ymin=0,ymax=1,alpha=.7,linestyle='--',color='r',
                lw=3,label='bminJ={0}'.format(bminJ))
                    
    plt.hold(False)
    plt.grid(True)
    plt.legend()
    plt.xlabel('b')
    plt.ylabel('Critere')
    plt.title('Critere =f(b)')
    
    #pdb.set_trace()
    
    
def test_estim_ecart_ang_opt_2():
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.sigma_v=0.0345
    param_MB.h_v=10
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    #param_MB.gammat=np.array([[1,1,1],[1,1,1],[1,1,1]])
    
    nb_echant=10
    data_synt=tom.TomoSARDataSet_synth(param_MB.Na,param_MB)
    W_k=data_synt.get_W_k_rect(param_MB,int(nb_echant),param_MB.Na)        
    
    hvJ,hvJ2,hvV,\
    sigJ,sigJ2,sigV,\
    gt_J,gt_J2,gt_V,\
    X_minJ,X_minJ2,X_minV,\
    minJ,minJ2,minV,\
    vec_b = e.estim_ecart_ang_opt2(W_k,param_MB)
    
    return hvJ,hvJ2,hvV,\
           sigJ,sigJ2,sigV,\
           gt_J,gt_J2,gt_V,\
           X_minJ,X_minJ2,X_minV,\
           minJ,minJ2,minV,\
           vec_b
           
def test_estim_ecart_ang_tot():
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB
    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.sigma_v=0.0345
    param_MB.h_v=30
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt(),Na)
    
    
    
    W_k=W_k_vrai_MB 
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param_MB.Na)
    
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param_MB.Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    #choix du a et b
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]
    b = (interv_b[0][0]+interv_b[0][1])/2
    b_vrai = tom.b_true(R_t,param_MB)
    _,Rg,Rv,Cg,Cv=tom.value_R_C(R_t,C_t,a,b)    
    
    vec_gt = np.array([param_MB.gammat[0,1],param_MB.gammat[0,2],param_MB.gammat[1,2]])
    
    q = e.estim_ecart_ang_tot(W_k,param_MB)
    print 'yo bvrai {0} hv_vrai {1}  sigvrai {2}'.format(b_vrai,param_MB.h_v,
                                                          param_MB.sigma_v)
    return 'vive les petits lapins' 

def test_estim_ecart_ang_scal():
    Np=3
    param=lp.load_param(name='DB_1')        
    W_k_vrai = tom.UPS_to_MPMB(param.get_upsilon_gt())
    data = tom.TomoSARDataSet_synth(param)
    Wnoise = data.get_W_k_rect(param,10**6)
    
    #W = W_k_vrai
    W = Wnoise
    W_norm,_ = tom.normalize_MPMB_PS_Tebald(W,param.Na)
    
    bvrai=tom.b_true_from_param(param)
    hv,sigv,vec_gt,bopt = e.estim_ecart_ang_scal(W,param,
                                                 critere='J2',
                                                 zg_connu=param.z_g,
                                                 U0=np.array([param.h_v,param.sigma_v]),
                                                 b0=bvrai)
    
    print 'b_vrai {0} b {1}'.format(bvrai,bopt)
    print 'hv_vrai {0} hv {1}'.format(param.h_v,hv)
    print 'sig_vrai {0} sig {1}'.format(param.sigma_v,sigv)
    print 'vec_gt_vrai {0}\n vec_gt {1}'.format(param.get_gtlist(),vec_gt)

    
def test_V_scal():
    """Test du criètre de vraissemblance V optimisé par rapport à b"""
    Np=3
    param=lp.load_param(name='DB_1')        
    #param.gammat=np.ones((3,3))
    Na = param.Na
    W_k_vrai = tom.UPS_to_MPMB(param.get_upsilon_gt())
    W_k = W_k_vrai
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param.Na)
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param.Na)    
    b_vrai = tom.b_true2(R_t,param)
    a_vrai = tom.a_true(R_t,param)
    param.N=1#CAS NON BRUITE
    N=param.N
 
    Crit=e.V_scal(b_vrai,W_k,R_t,C_t,param.get_kzlist(),\
             np.cos(param.theta),a_vrai,Na,N,param.get_gtlist())        
    
    hv,sigv,bopt = e.estim_V_scal(W_k,param)

    N=param.N#Taille d'echant
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W_k,Na)
    Ups = tom.MPMB_to_UPS(W_k)
    
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)
    
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    bvrai=tom.b_true2(R_t,param)        
    bmin=np.min(interv_b[0])
    bmax=np.max(interv_b[0])

    Nbpts=50
    CritV=np.zeros((Nbpts,1))    
    vecb=np.linspace(bmin,bmax,Nbpts)
    for i,b in enumerate(vecb):
        CritV[i]=e.V_scal(b,W_k,R_t,C_t,vec_kz,costheta,a,Na,N,vec_gt)    
        
    plt.plot(vecb,CritV)
    plt.axvline(bmin,hold=True,color='k')
    plt.axvline(bmax,hold=True,color='k')
    plt.axvline(bvrai,hold=True,color='g')
    plt.axvline(bopt,hold=True,color='r')
    plt.show()
    print 'bmin {0} bmax {1} bvrai {2} bopt {3}'.format(bmin,bmax,bvrai,bopt)
    gv=param.get_gamma_v()
    gvgt=param.get_gamma_v_gt()
    ga,gbopt=tom.gamma_a_b(R_t,a,bopt)
    ga,gbvrai=tom.gamma_a_b(R_t,a,bvrai)
    pt.plot_cu_sm_possible2(Ups,R_t,interv_a,interv_b,0,1)
    plt.hold(True)
    plt.plot(np.angle(gbvrai),np.abs(gbvrai),'ok',markersize=5)
    plt.plot(np.angle(gv),np.abs(gv),'sg',markersize=4)
    plt.plot(np.angle(gvgt),np.abs(gvgt),'sg',markersize=4)
    
    plt.plot(np.angle(gbopt),np.abs(gbopt),'r^',markersize=4)

def test_V_scal_gt_inconnu():
    Np=3
    param=lp.load_param(name='DB_1')
    param.z_g=0
    #param.gammat=np.ones((3,3))
    Na = param.Na
    W_k_vrai = tom.UPS_to_MPMB(param.get_upsilon_gt())
    W_k = W_k_vrai
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param.Na)
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param.Na)    
    b_vrai = tom.b_true2(R_t,param)
    a_vrai = tom.a_true(R_t,param)
    param.N=1#CAS NON BRUITE
    N=param.N
 
#    Crit=e.V_scal_gt_inconnu(b_vrai,W_k,R_t,C_t,param.get_kzlist(),\
#             np.cos(param.theta),a_vrai,Na,N,param.z_g)        
    
    hv,sigv,vec_gt,bopt = e.estim_V_scal_gt_inconnu(W_k,param,zg_connu=param.z_g)

    N=param.N#Taille d'echant
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W_k,Na)
    Ups = tom.MPMB_to_UPS(W_k)
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    Nbb=150
    Crit=np.zeros(Nbb)
    vec_b=np.linspace(b_vrai-b_vrai*0.05,b_vrai+b_vrai*0.05,Nbb)
    for idxb,b in enumerate(vec_b):
        Crit[idxb]=e.V_scal_gt_inconnu(b,W_k,R_t,C_t,param.get_kzlist(),\
             np.cos(param.theta),a_vrai,Na,N,param.z_g)
    
    plt.plot(vec_b,Crit) 
    plt.vlines(np.argmin(np.abs(vec_b-b_vrai)),'r')
    #plt.vlines(np.argmin(np.abs(vec_b-b_opt)),'r')
    plt.yscale('log')
    
    bvrai=tom.b_true2(R_t,param)        
    bmin=np.min(interv_b[0])
    bmax=np.max(interv_b[0])
    pdb.set_trace()

def test_mlogV_gt_inconnu():
    """Est ce que le critere mlogV_gt_inconnu atteint son min pour thetavrai?
    test sur plusieurs tirage autour la val vrai"""
    
    P=10000
    Crit = np.zeros(P)
    param=lp.load_param('DB_1')
    theta_vrai=np.array([30,0.0345,0.7,0.8,0.8])
    param.N=1
    
    for i in range(P):
        eps=np.random.randn(5)*0.1
        Crit[i]=e.mlogV_gt_inconnu(theta_vrai+eps,
                                    param.get_upsilon_gt(),
                                    param.get_kzlist(),np.cos(param.theta),
                                    param.Na,param.get_I1()*param.T_vol,
                                    param.get_a()*param.T_ground,
                                    param.N,param.z_g)
                                    
    V_vrai = e.mlogV_gt_inconnu(theta_vrai,
                                    param.get_upsilon_gt(),
                                    param.get_kzlist(),np.cos(param.theta),
                                    param.Na,param.get_I1()*param.T_vol,
                                    param.get_a()*param.T_ground,
                                    param.N,param.z_g)
    
    #pdb.set_trace()
    plt.hist(Crit[np.isfinite(Crit)],bins=80)
    plt.hold(True)
    print V_vrai
    
    plt.axvline(V_vrai,color='r')
    
def test_mlogV_gt_inconnu_all():
    
    P=10000
    Crit = np.zeros(P)
    param=lp.load_param('DB_1');param.z_g=0
    #tom.b_true2(1.044)
    Np=3
    param.N=1
    W=tom.UPS_to_MPMB(param.get_upsilon_gt())
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W,param.Na)
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param.Na)    
    b_vrai = tom.b_true2(R_t,param)
    a_vrai = tom.a_true(R_t,param)
    theta_vrai=np.array([b_vrai,30,0.0345,0.7,0.8,0.8])
    
    for i in range(P):
        eps=np.random.randn(6)*0.01
        Crit[i]=e.mlogV_gt_inconnu_all(theta_vrai+eps,
                                        W,R_t,C_t,
                                        param.get_kzlist(),np.cos(param.theta),a_vrai,
                                        param.Na,                                  
                                        param.N,param.z_g)
                                    
    V_vrai = e.mlogV_gt_inconnu_all(theta_vrai,
                                        W,R_t,C_t,
                                        param.get_kzlist(),np.cos(param.theta),a_vrai,
                                        param.Na,                                    
                                        param.N,param.z_g)
    CC=Crit[np.isfinite(Crit)]                                    
    plt.hist(CC,bins=200,range=(np.min(CC)-5,50))
    plt.hold(True)
    print V_vrai
    plt.axvline(V_vrai,color='r')
    pdb.set_trace()
    
def test_mlogV_Tg_Tv_known():
    P=10000
    Crit = np.zeros(P)
    param=lp.load_param('DB_1');param.z_g=0
    #tom.b_true2(1.044)
    Np=3
    param.N=1
    Ups=param.get_upsilon_gt()
    theta_vrai=np.array([30,0.0345,0.7,0.8,0.8])
    
    for i in range(P):
        eps=np.random.randn(5)*1
        Crit[i]=e.mlogV_Tg_Tv_known(theta_vrai+eps,
                                        Ups,
                                        param.get_kzlist(),np.cos(param.theta),
                                        param.Na,param.T_vol,param.T_ground,
                                        param.N,param.z_g)
                                    
    V_vrai = e.mlogV_Tg_Tv_known(theta_vrai,
                                    Ups,
                                    param.get_kzlist(),np.cos(param.theta),
                                    param.Na,param.T_vol,param.T_ground,
                                    param.N,param.z_g)
                                    
    CC=Crit[np.isfinite(Crit)]                                    
    plt.hist(CC,bins=200,range=(np.min(CC)-5,300))
    plt.hold(True)
    print V_vrai
    plt.axvline(V_vrai,color='r')
    pdb.set_trace()
    
def test_estim_mlogV_Tg_Tv_known():
    print '---- test_estim_mlogV_Tg_Tv_known'
    param=lp.load_param(name='DB_1')        
    param.N = 1#A mettre si non bruité
    Ups = param.get_upsilon_gt()
    UU0=np.hstack((np.array([param.h_v,param.sigma_v]),param.get_gtlist()))  
    hv,sigv,vec_gt = e.estim_mlogV_Tg_Tv_known(Ups,param,zg_connu=param.z_g,
                                               U0=UU0)

    print 'hv_vrai {0} hv {1}'.format(param.h_v,hv)
    print 'sig_vrai {0} sig {1}'.format(param.sigma_v,sigv)
    print 'vec_gt_vrai {0}\n vec_gt {1}'.format(param.get_gtlist(),vec_gt)
    
def test_estim_V_scal():
    print '--- test Estim_V_scal ---- '
    Np=3
    param_MB=lp.load_param(name='DB_1')        
    param_MB.N=1#A mettre si non bruité
    #param_MB.z_g=0
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt())
    data_synt = tom.TomoSARDataSet_synth(param_MB)    
    W = data_synt.get_W_k_rect(param_MB,int(param_MB.N))    
    
    W_k = W
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param_MB.Na)
    W_k_norm = tom.retranch_phig_W(param_MB.z_g,W_k_norm,param_MB.get_kzlist())
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param_MB.Na)    
    
    #Bidouille infernale car sinon le bvrai n'est pas le bopt 
    #car dans l'estimateur on enleve la phase du sol.
    zg_temp=param_MB.z_g;
    param_MB.z_g=0;
    b_vrai = tom.b_true_from_param(param_MB)
    param_MB.z_g=zg_temp
    
    hv,sigv,bopt = e.estim_V_scal(W_k,param_MB,zg_connu=param_MB.z_g,
                                  U0=np.array([param_MB.h_v,param_MB.sigma_v]),
                                  critere='J2')

    print 'b_vrai {0}  bopt {1}'.format(b_vrai,bopt)
    print 'hv_vrai {0} hv {1}'.format(param_MB.h_v,hv) 
    print 'sig_vrai {0} sig {1}'.format(param_MB.sigma_v,sigv)

def test_estim_V_scal_gt_inconnu():
    Np=3
    param_MB=lp.load_param(name='DB_1')        
    param_MB.N=1#A mettre si non bruité
    param_MB.z_g=0
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt())
    W_k = W_k_vrai_MB 
    
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param_MB.Na)
    W_k_norm = tom.retranch_phig_W(param_MB.z_g,W_k_norm,param_MB.get_kzlist())
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param_MB.Na)    
    
    hv,sigv,vec_gt,bopt = e.estim_V_scal_gt_inconnu(W_k,param_MB,
                                                    zg_connu=param_MB.z_g)
    b_vrai = tom.b_true2(R_t,param_MB)                                                   
    #pt.plot_rc_mb(tom.MPMB_to_UPS(W_k_norm))
    print 'b_vrai {0}  bopt {1}'.format(b_vrai,bopt)
    print 'hv_vrai {0} hv {1}'.format(param_MB.h_v,hv) 
    print 'sig_vrai {0} sig {1}'.format(param_MB.sigma_v,sigv)
    print 'vec_gt gt12: {0} gt13: {1} gt23: {2}'.format(vec_gt[0],vec_gt[1],vec_gt[2])
    
def estim_mlogV_gt_inconnu_all():
    Np=3
    param_MB=lp.load_param(name='DB_1')        
    param_MB.N=1#A mettre si non bruité
    param_MB.z_g=0
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt())
    W_k = W_k_vrai_MB 
    
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k,param_MB.Na)
    W_k_norm = tom.retranch_phig_W(param_MB.z_g,W_k_norm,param_MB.get_kzlist())
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param_MB.Na)    
    
    hv,sigv,vec_gt,bopt = e.estim_mlogV_gt_inconnu_all(W_k,param_MB,
                                                    zg_connu=param_MB.z_g)
    b_vrai = tom.b_true2(R_t,param_MB)                                                   
    #pt.plot_rc_mb(tom.MPMB_to_UPS(W_k_norm))
    print 'b_vrai {0}  bopt {1}'.format(b_vrai,bopt)
    print 'hv_vrai {0} hv {1}'.format(param_MB.h_v,hv) 
    print 'sig_vrai {0} sig {1}'.format(param_MB.sigma_v,sigv)
    print 'vec_gt gt12: {0} gt13: {1} gt23: {2}'.format(vec_gt[0],vec_gt[1],vec_gt[2])
    
def test_montecarl_estim_ecart_ang_opt():    
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.sigma_v=0.0345
    param_MB.h_v=30
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    #param_MB.gammat=np.array([[1,1,1],[1,1,1],[1,1,1]])
    estim=1
    if estim:            
        hvJ,hvJ2,hvV,\
        sigJ,sigJ2,sigV,\
        vec_gt_J,vec_gt_J2,vec_gt_V,\
        meanhvJ,meanhvJ2,meanhvV,\
        varhvJ,varhvJ2,varhvV,\
        meansigJ,meansigJ2,meansigV,\
        varsigJ,varsigJ2,varsigV,\
        meangtJ,meangtJ2,meangtV,\
        vargtJ,vargtJ2,vargtV,\
        vec_N,P = e.monte_carl_estim_ecart_ang_opt(param_MB)    
                                
    save=0                              
    path='D:/PIERRE CAPDESSUS/Python/Code_Pierre/monte_carl_estim_ecar_ang_opt/Test3/'
    
    list_of_save= ['hvJ','hvJ2','hvV','sigJ','sigJ2','sigV',\
                  'vec_gt_J','vec_gt_J2','vec_gt_V',\
                  'meanhvJ','meanhvJ2','meanhvV',\
                  'varhvJ','varhvJ2','varhvV',\
                  'meansigJ','meansigJ2','meansigV',\
                  'varsigJ','varsigJ2','varsigV',\
                  'meangtJ','meangtJ2','meangtV',\
                  'vargtJ','vargtJ2','vargtV',\
                  'vec_N','P']
                  
    list_of_save_fig=['fig_moyJ','fig_varJ','fig_moyJ2','fig_varJ2',\
                      'fig_moyV','fig_varV']    
    if save:       
        for idx,var in enumerate(list_of_save):
            np.save(path+var,eval(var))
        for idx,fig in enumerate(list_of_save_fig):
            #pdb.set_trace()
            eval(fig).savefig(path+fig+'.png',format='png',\
                        bbox_inches='tight',pad_inches=1)

    load=0    
    if load:        
            P=np.load(path+'P'+'.npy')
            vec_N=np.load(path+'vec_N'+'.npy')
            meanhvJ=np.load(path+'meanhvJ'+'.npy')
            varhvJ=np.load(path+'varhvJ'+'.npy')
            meansigJ=np.load(path+'meansigJ'+'.npy')
            varsigJ=np.load(path+'varsigJ'+'.npy')
            meanhvJ2=np.load(path+'meanhvJ2'+'.npy')
            varhvJ2=np.load(path+'varhvJ2'+'.npy')
            meansigJ2=np.load(path+'meansigJ2'+'.npy')
            varsigJ2=np.load(path+'varsigJ2'+'.npy')

            
            
    #pdb.set_trace()
    """
    pt.plot_biais_variance_err(vec_N,meanhvJ,varhvJ,P,\
                                'J hv moyen ','J Var hv',param_MB.h_v,'hv')
    pt.plot_biais_variance_err(vec_N,meanhvJ2,varhvJ2,P,\
                                'J2 hv moyen ','J2 Var hv',param_MB.h_v,'hv')
    pt.plot_biais_variance_err(vec_N,meansigJ,varsigJ,P,\
                                'J sigma moyen ','J Var sig',param_MB.sigma_v,'sigma')
    pt.plot_biais_variance_err(vec_N,meansigJ2,varsigJ2,P,\
                                'J2 sigma moyen ','J2 Var sig',param_MB.sigma_v,'sigma')
                                
    
    f_moyJ,f_varJ=pt.plot_biais_variance_err(vec_N,meanhvJ,varhvJ,P,\
                                'J hv moyen ','J Var hv',param_MB.h_v,'hv')
    fig_moyJ2,fig_varJ2=pt.plot_biais_variance_err(vec_N,meanhvJ2,varhvJ2,P,\
                                'J2 hv moyen ','J2 Var hv',param_MB.h_v,'hv',f_moyJ,f_varJ)
    """                            
    meanhv=np.vstack((meanhvJ,meanhvJ2))
    varhv=np.vstack((varhvJ,varhvJ2))
    meansig=np.vstack((meansigJ,meansigJ2))
    varsig=np.vstack((varsigJ,varsigJ2))
    
    pt.plot_biais_variance_err_mutl(vec_N,meanhv,varhv,P,\
                            'hv moyen','variance hv',param_MB.h_v,'hv',
                            ['J','J2'],['J','J2'])
                            
    pt.plot_biais_variance_err_mutl(vec_N,meansig,varsig,P,\
                            'sig moyen','variance sig',param_MB.sigma_v,'sig',
                            ['J1','J2'],['J','J2'])
    
    """                                
    fig_moyV,fig_varV=pt.plot_biais_variance_err(vec_N,meanhvV,varhvV,P,\
                                'V hv moyen ','V Var hv',param_MB.h_v)
    """
        
    return     hvJ,hvJ2,hvV,\
               sigJ,sigJ2,sigV,\
               vec_gt_J,vec_gt_J2,vec_gt_V,\
               meanhvJ,meanhvJ2,meanhvV,\
               varhvJ,varhvJ2,varhvV,\
               meansigJ,meansigJ2,meansigV,\
               varsigJ,varsigJ2,varsigV,\
               meangtJ,meangtJ2,meangtV,\
               vargtJ,vargtJ2,vargtV
               
def test_montecarl_estim_ecart_ang_opt2():
    
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.sigma_v=0.0345
    param_MB.h_v=30
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    #param_MB.gammat=np.array([[1,1,1],[1,1,1],[1,1,1]])
    hvJ,hvJ2,hvV,\
    sigJ,sigJ2,sigV,\
    vec_gt_J,vec_gt_J2,vec_gt_V,\
    meanhvJ,meanhvJ2,meanhvV,\
    varhvJ,varhvJ2,varhvV,\
    meansigJ,meansigJ2,meansigV,\
    varsigJ,varsigJ2,varsigV,\
    meangtJ,meangtJ2,meangtV,\
    vargtJ,vargtJ2,vargtV,\
    vec_N,P = e.monte_carl_estim_ecart_ang_opt2(param_MB)    
    
    plt.figure()
    plt.plot(hvJ)
    plt.figure()
    plt.plot(sigJ)
    print meanhvJ
    print np.sqrt(varhvJ)
    
    fig_moyJ,fig_varJ=pt.plot_biais_variance_err(vec_N,meanhvJ,varhvJ,P,\
                                'J hv moyen ','J Var hv',param_MB.h_v)
    fig_moyJ2,fig_varJ2=pt.plot_biais_variance_err(vec_N,meanhvJ2,varhvJ2,P,\
                                'J2 hv moyen ','J2 Var hv',param_MB.h_v)
    fig_moyV,fig_varV=pt.plot_biais_variance_err(vec_N,meanhvV,varhvV,P,\
                                'V hv moyen ','V Var hv',param_MB.h_v)
    
    print 'P ={0} N={1}'.format(P,vec_N)
    print 'hvJ,hvJ2,hvV,sigJ,sigJ2,sigV'
    print  meanhvJ,meanhvJ2,meanhvV,meansigJ,meansigJ2,meansigV
    print '------------------ std ----------------------------'
    print np.sqrt(np.array([varhvJ,varhvJ2,varhvV,varsigJ,varsigJ2,varsigV]))
    
    
    save=1         
                     
    if save:       
        path='D:/PIERRE CAPDESSUS/Python/Code_Pierre/monte_carl_estim_ecar_ang_opt2/Test2/'
        
        list_of_save= ['hvJ','hvJ2','hvV','sigJ','sigJ2','sigV',\
                      'vec_gt_J','vec_gt_J2','vec_gt_V',\
                      'meanhvJ','meanhvJ2','meanhvV',\
                      'varhvJ','varhvJ2','varhvV',\
                      'meansigJ','meansigJ2','meansigV',\
                      'varsigJ','varsigJ2','varsigV',\
                      'meangtJ','meangtJ2','meangtV',\
                      'vargtJ','vargtJ2','vargtV',\
                      'vec_N','P']
                      
        list_of_save_fig=['fig_moyJ','fig_varJ','fig_moyJ2','fig_varJ2',\
                          'fig_moyV','fig_varV']
        for idx,var in enumerate(list_of_save):
            np.save(path+var,eval(var))
        for idx,fig in enumerate(list_of_save_fig):
            #pdb.set_trace()
            eval(fig).savefig(path+fig+'.png',format='png',\
                        bbox_inches='tight',pad_inches=1)
        
    #pdb.set_trace()
    return     hvJ,hvJ2,hvV,\
               sigJ,sigJ2,sigV,\
               vec_gt_J,vec_gt_J2,vec_gt_V,\
               meanhvJ,meanhvJ2,meanhvV,\
               varhvJ,varhvJ2,varhvV,\
               meansigJ,meansigJ2,meansigV,\
               varsigJ,varsigJ2,varsigV,\
               meangtJ,meangtJ2,meangtV,\
               vargtJ,vargtJ2,vargtV
               
def test_monte_carlo_estim_ang_scal():
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    X=0.2
    k_z = [0.1,0.15]
    savdir='D:\PIERRE CAPDESSUS\Resultats_simu\Monte_carlo'
    #MB    
    A=0.95
    E=200
    X=0.2
    Tvol,Tground=mb.Tv_Tg_from_A_E_X(A,E,X)
    gamma_t_mat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    param_MB=mb.param_rvog(N=100,k_z=[0.1,0.15],theta=45*np.pi/180,T_vol=Tvol,
                    T_ground=Tground,h_v=30,z_g=0,sigma_v=0.0345,
                    mat_gamma_t=gamma_t_mat)

    
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_simu={}
    param_simu['P']=5
    param_simu['NbN']=1
    param_simu['Nmin']=100
    param_simu['Nmax']=1000
    
    mat_hvJ,mat_sigv,\
    vec_meanhv,vec_meansigv,\
    vec_varhv,vec_varsigv,\
    vec_N = e.mont_carl_estim_ang_scal(param_MB,param_simu)
    
    savedata=1    
    if savedata:
        list_of_save=['mat_hvJ','mat_sigv','vec_meanhv','vec_meansigv',
                      'vec_varhv','vec_varsigv','P','NbN','Nmin','Nmax']
        for idx,var in enumerate(list_of_save):
            np.save(savdir+'\\'+var)
    
    np.semilogy(vec_N,np.sqrt(vec_varhv))
               
            
def plot_gamma_v():
    
    A=0.8;E=150;
    Na=2
    Np=3
    param = mb.param_rvog(Na)
    param = mb.rvog_reduction(param,A,E)
    tropi = tom.TomoSARDataSet_synth(Na,param)
    
    param1 = copy(param)
    param1.k_z=[0.1]
    param2 = copy(param)
    param2.k_z=[0.2]
    W_k_vrai1 = tom.UPS_to_MPMB(param1.get_upsilon(),Na)
    W_k_norm_vrai1 = tom.normalize_MPMB_PS_Tebald(W_k_vrai1,Na)
    mat_covariance_vrai1 = tom.MPMB_to_UPS(W_k_vrai1,Na)
    
    R_t1,C_t1,_ = tom.sm_separation(W_k_norm_vrai1,Np,Na)     
    interv_a1,interv_b1,Cond1,alpha1 =\
       tom.search_space_definition(R_t1,C_t1,Na)  

    W_k_vrai2 = tom.UPS_to_MPMB(param2.get_upsilon(),Na)
    W_k_norm_vrai2 = tom.normalize_MPMB_PS_Tebald(W_k_vrai2,Na)
    mat_covariance_vrai2 = tom.MPMB_to_UPS(W_k_vrai2,Na)
    
    R_t2,C_t2,_ = tom.sm_separation(W_k_norm_vrai2,Np,Na)     
    interv_a2,interv_b2,Cond2,alpha2 =\
       tom.search_space_definition(R_t2,C_t2,Na)  

    Npts=50
    #vec_k_z = np.linspace(0.1,(2*np.pi)/param.h_v,Npts)
    vec_k_z = np.linspace(0.1,0.2,Npts)
    vec_gamma_v = np.zeros((Npts,1),dtype='complex')
    
    for idx,k_z in enumerate(vec_k_z):
        param.k_z=[k_z]
        vec_gamma_v[idx]=param.get_gamma_v()
    
    plt.close('all')    
    fig1,ax=pt.plot_cu_sm_possible(tom.MPMB_to_UPS(W_k_vrai1,2),R_t1,\
                            interv_a1,interv_b1,0,1,'Region coherence')    
    fig2,ax=pt.plot_cu_sm_possible(tom.MPMB_to_UPS(W_k_vrai2,2),R_t2,\
                            interv_a2,interv_b2,0,1,'Region coherence',fig1)  

    omega_blanc1 = tom.polinsar_compute_omega12blanchi(param1.get_upsilon())
    theta1,_ = tom.polinsar_estime_droite(omega_blanc1)
    omega_blanc2 = tom.polinsar_compute_omega12blanchi(param2.get_upsilon())
    theta2,_ = tom.polinsar_estime_droite(omega_blanc2)
    """
    angle_rot = theta1-theta2
    print angle_rot*180/np.pi    
    centre = 1
    Ups_rot,R_t_rot=bl.rot_omega_sm(centre,angle_rot,param2.get_upsilon(),R_t2)
    
    fig3,ax=pt.plot_cu_sm_possible(Ups_rot,R_t_rot,\
                            interv_a2,interv_b2,0,1,'Region coherence',fig2)                                
    """                              
    
    ax.hold(True)                     
    ax.plot(np.angle(vec_gamma_v),np.abs(vec_gamma_v),'--')      
    
    ind = [0,-1]
    ax.plot(np.angle(vec_gamma_v[ind]),np.abs(vec_gamma_v[ind]),'bo')      
    plt.annotate('kz='+str(vec_k_z[0]),\
                    xy=(np.angle(vec_gamma_v[0]),np.abs(vec_gamma_v[0])),\
                    xytext=(np.angle(vec_gamma_v[0]-np.pi/10),np.abs(vec_gamma_v[0]+0.1)),\
                    arrowprops=dict(facecolor='black',shrink=0.05)
                    )
    plt.annotate('kz='+str(vec_k_z[-1]),\
                    xy=(np.angle(vec_gamma_v[-1]),np.abs(vec_gamma_v[-1])),\
                    xytext=(np.angle(vec_gamma_v[-1]-np.pi/10),np.abs(vec_gamma_v[-1]+0.1)),\
                    arrowprops=dict(facecolor='black',shrink=0.05)
                    )

def plot_vp_chgtm_Na():
    Na = 3
    Np = 3
    A = 0.95
    E = 71
    k_z = [0.1,0.2]

    #Test avec matrice sans bruit de speckle
    #SB
    param_SB = mb.param_rvog(2)
    param_SB = mb.rvog_reduction(param_SB,A,E)
    param_SB.k_z = [k_z[0]]
    param_SB.z_g = 0
    W_k_vrai_SB = tom.UPS_to_MPMB(param_SB.get_upsilon(),2)
    
    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z = k_z
    param_MB.z_g = 0
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon(),Na)

    #sm_separation SB
    R_t_SB,C_t_SB,_ = tom.sm_separation(W_k_vrai_SB,Np,2)             
    interv_a_SB,interv_b_SB,Cond_SB,alpha_SB =\
        tom.search_space_definition(R_t_SB,C_t_SB,2)  
    #sm_separation MB
    R_t_MB,C_t_MB,_ = tom.sm_separation(W_k_vrai_MB,Np,Na)             
    interv_a_MB,interv_b_MB,Cond_MB,alpha_MB =\
        tom.search_space_definition(R_t_MB,C_t_MB,Na) 

    """        
    plt.close('all')
    pt.plot_vp_Rv_Cg(R_t_SB,C_t_SB,interv_a_SB,interv_b_SB,alpha_SB)
    pt.plot_vp_Rv_Cg(R_t_MB,C_t_MB,interv_a_MB,interv_b_MB,alpha_MB)
    """

    plt.close('all')
    #pt.plot_vp_Rv_Cg_nodiag(R_t_SB,C_t_SB,interv_a_SB,interv_b_SB,alpha_SB)
    pt.plot_vp_Rv_Cg_nodiag(R_t_MB,C_t_MB,interv_a_MB,interv_b_MB,alpha_MB)
    
    
    print 'interv_b_SB',interv_b_SB[0][0],interv_b_SB[0][1]
    a_SB = interv_a_SB[0][0]   
    a_MB = interv_a_MB[0][0]   
    _,g_min_sb = tom.gamma_a_b(R_t_SB,a_SB,interv_b_SB[0][0])
    _,g_max_sb = tom.gamma_a_b(R_t_SB,a_SB,interv_b_SB[0][1])
    _,g_min_mb = tom.gamma_a_b(R_t_MB,a_MB,interv_b_MB[0][0])
    _,g_max_mb = tom.gamma_a_b(R_t_MB,a_MB,interv_b_MB[0][1])
        
    print 'gamma_vol_SB',g_min_sb,g_max_sb        
    print 'interv_b_MB',interv_b_MB[0][0],interv_b_MB[0][1]
    print 'gamma_vol_MB',g_min_mb,g_max_mb
    
    print " ༼ つ ◕_◕ ༽つ : rapport r=g13/g12g23 pour le volume"
    
    print 'angle r={0}'.format(np.angle(R_t_MB[1][0,2]/R_t_MB[1][0,1]**2)*180/np.pi)
    print 'abs r={0}'.format(np.abs(R_t_MB[1][0,2]/R_t_MB[1][0,1]**2))
    
def variation_rac_pdp():
    
    Na = 3
    Np = 3
    A = 0.95
    E = 71
    k_z = [0.1,0.2]

    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z = k_z
    param_MB.z_g = 0
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon(),Na)
    
    list_kz=[[0.1,0.2],[0.1,0.3],[0.2,0.3],[0.2,0.4]] #chque element de liste: (kz12,kz13)
    #print 'h_amb = {0}',2*np.pi*[1/kz for kz in list_kz]
    print 'hamb={0}'.format([2*np.pi/kz for kzMB in list_kz for kz in  kzMB])
    nb_pts = 100
    
    #vec_hv = np.linspace(5,2*np.pi*1/np.max(list_kz),nb_pts)
    vec_hv = np.linspace(5,30,nb_pts)
    vec_r2_hv = np.zeros((len(list_kz),nb_pts))
    ratio = np.zeros((len(list_kz),nb_pts))
    
    for idx_kz in range(len(list_kz)):
        param_MB.k_z = list_kz[idx_kz]
        for idx_hv,h_v in enumerate(vec_hv):        
            param_MB.h_v = h_v 
            gammav12=param_MB.get_gamma_v(0,1)
            gammav13=param_MB.get_gamma_v(0,2)        
            ratio[idx_kz,idx_hv] = gammav13/gammav12**2
            _,_,_,vec_r2_hv[idx_kz,idx_hv],_,_, = tom.pol_pdp(ratio[idx_kz,idx_hv])
    plt.close('all')            
    plt.figure()    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
    plot_obj = plt.plot(vec_hv,vec_r2_hv.T,lw='5')
    plt.legend(iter(plot_obj),[ 'kz='+str(kz) for kz in list_kz],loc=3)
    plt.grid(True)
    plt.xlabel('hv',fontsize=18)
    plt.ylabel('racine1 P',fontsize=18)
    plt.title('Racine de P pour differentes configurations')
    matplotlib.rc('font', **font)
    
    plt.figure()
    plt.plot(vec_hv,np.abs(ratio).T)
    plt.xlabel('hv',fontsize=26)
    plt.ylabel('racine1 P',fontsize=16)

def test_selection_ground_selection_MB():
    Na = 2
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.12]
    #MB
    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=[k_z[0]]
    param_MB.h_v=10
    param_MB.z_g =0
    #param_MB.T_ground=np.array([[110,49+15*1j,0],[49-15*1j,110,0],[0,0,79]])
    
    #param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt(),Na)
    #Matrice bruitée
    nb_echant=10000
    data=tom.TomoSARDataSet_synth(param_MB.Na,param_MB)
    W_k_norm_br=data.get_W_k_norm_rect(param_MB,nb_echant,param_MB.Na)
    #W_k_br=data.get_W_k_rect(param_MB,nb_echant,param_MB.Na)
    #Ups_br=tom.MPMB_to_UPS(W_k_br,param_MB.Na)    
    
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k_vrai_MB,Na)
    
    Ups_norm_br = tom.MPMB_to_UPS(W_k_norm_br,Na)
    
               
    R_t,C_t,_ = tom.sm_separation(W_k_norm_br,Np,Na)
    interv_a,interv_b,mat_cond,_ = tom.search_space_definition(R_t,C_t,Na)

    ant1,ant2=(0,1)
    interv_a_good,interv_b_good=tom.ground_selection_MB_new(R_t,interv_a,interv_b)
    
    fig1,_=pt.plot_cu_sm_possible(Ups_norm_br,R_t,interv_a_good,interv_b_good,
                                  ant1,ant2,'Test sol')
    fig2,_=pt.plot_cu_sm_possible(Ups_norm_br,R_t,interv_a_good,interv_b_good,
                                  ant1,ant2,'Test sol',fig=fig1)
    plt.draw()
    
    fig1,_=pt.plot_cu_sm_possible(Ups_norm_br,R_t,interv_a,interv_b,
                                  ant1,ant2,'Test sol')
    fig2,_=pt.plot_cu_sm_possible(Ups_norm_br,R_t,interv_a,interv_b,
                                  ant1,ant2,'Test sol',fig=fig1)
    plt.draw()
    print interv_a_good,interv_a
    print interv_b_good,interv_b
    #print Ups_norm
    #pdb.set_trace()
    
def test_reconstruction_UPS():

    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB    
    param_MB = mb.param_rvog(Na)
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.sigma_v=0.0345
    param_MB.h_v=30.
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    #param_MB.gammat=np.array([[1,0.8,0.9],[1,1,0.7],[1,1,1]])
    param_MB.gammat=np.array([[1,0.8,0.6],[1,1,0.7],[1,1,1]])
    #param_MB.gammat=[1,1]
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt(),Na)
    W_k_norm,E = tom.normalize_MPMB_PS_Tebald(W_k_vrai_MB,Na)
    Ups = param_MB.get_upsilon_gt()

    nbpth = 401
    nbpts = 400
    vec_hv = np.logspace(np.log10(5),np.log10(40),nbpth)
    vec_sig = np.logspace(np.log10(0.01),
                          np.log10(0.05),nbpts)
                              
    Ressemb = np.zeros((nbpth,nbpts),dtype='float')
    EQM = np.zeros((nbpth,nbpts),dtype='float')
    EQM_temp = np.zeros((nbpth,nbpts),dtype='float')
    V = np.zeros((nbpth,nbpts),dtype='float')
    V_temp = np.zeros((nbpth,nbpts),dtype='float')
    
    for i,hv in enumerate(vec_hv):
        for j,sig in enumerate(vec_sig):        
            p_temp=copy(param_MB)
            p_temp.sigma_v=sig
            p_temp.h_v=hv            
            Ups_recon_temp=p_temp.get_upsilon_gt()
            Ups_recon =e.Ups_reconstr2(hv,sig,
                                       param_MB.get_gtlist(),
                                       param_MB.get_kzlist(),
                                       np.cos(param_MB.theta),
                                       param_MB.Na,param_MB.T_vol,
                                       param_MB.T_ground)
                                       
            Ups_est = np.copy(Ups)
            Ups_mod = np.copy(Ups_recon)
            N=1                    
            V[i,j]=e.vraissemb(Ups_est,Ups_mod,N)            
            V_temp[i,j]=e.vraissemb(Ups_est,Ups_recon_temp,N)            
            
            diff = Ups_est-Ups_mod
            diff_temp = Ups_est-Ups_recon_temp
            EQM[i,j]=np.trace(diff.dot(diff.T.conj()))
            EQM_temp[i,j]=np.trace(diff_temp.dot(diff_temp.T.conj()))
            if i == 100 and j==200:
                pdb.set_trace()
            
    idxEQM=zip(*np.where(EQM==np.nanmin(EQM)))
    idxEQM_temp=zip(*np.where(EQM_temp==np.nanmin(EQM_temp)))
    idxV=zip(*np.where(V==np.nanmin(V)))
    idxV_temp=zip(*np.where(V_temp==np.nanmin(V_temp)))
    
    idxhv_vrai=np.argmin(np.abs(vec_hv-param_MB.h_v))
    idxsig_vrai=np.argmin(np.abs(vec_sig-param_MB.sigma_v))

    if idxV[0][0] != idxhv_vrai:
        print 'Vraissemblance : ça pue pour hv'
    else:
        print 'Vraissemblance : atteint bonne valeur hv'
        
    if idxV[0][1] != idxsig_vrai:
        print 'Vraissemblance : ça pue pour sig'
    else:
        print 'Vraissemblance : atteint bonne valeur sig'
        
    if idxEQM[0][0] != idxhv_vrai:
        print 'EQM : ça pue pour hv'
    else:
        print 'EQM : atteint bonne valeur hv'
        
    if idxEQM[0][1] != idxsig_vrai:
        print 'EQM : ça pue pour sig'
    else:
        print 'EQM : atteint bonne valeur sig'
    
    
    plt.close('all')
    plt.figure()
    plt.imshow(EQM,origin='lower')
    plt.plot(idxEQM[0][1],idxEQM[0][0],'wo')
    plt.title='EQM'
    plt.axis('tight')
    plt.colorbar()
    
    plt.figure()
    plt.imshow(EQM_temp,origin='lower')
    plt.plot(idxEQM_temp[0][1],idxEQM_temp[0][0],'wo')
    plt.title='EQM'
    plt.axis('tight')
    plt.colorbar()
    """
    plt.figure()
    plt.imshow(np.log10(V),origin='lower')
    plt.plot(idxV[0][1],idxV[0][0],'wo')
    plt.axis('tight')
    plt.colorbar()
    plt.title='V'
    """
    #plt.colorbar()
    print vec_hv[idxEQM[0][0]]
    print vec_sig[idxEQM[0][1]]
    print vec_hv[idxV[0][0]]
    print vec_sig[idxV[0][1]]
 
def test_estim_Tvol_Tground_from_W():
    
    np.set_printoptions(precision=3)
    param = lp.load_param('DB_1')
    W = tom.UPS_to_MPMB(param.get_upsilon_gt())    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,param.Na)

    R_t,C_t,_ = tom.sm_separation(W_norm,Np=3,Na=param.Na)        
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,param.Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    #choix du a et b
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))][0]
    
    bvrai = tom.b_true2(R_t,param)
    avrai = tom.a_true(R_t,param)
    
    
    
    _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,avrai,bvrai)
    aTground,I1Tvol=tom.denormalisation_teb(W,param.Na,Cg,Cv)
    Ups_reconstr=e.Ups_reconstr2(param.h_v,param.sigma_v,param.get_gtlist(),
                    param.get_kzlist(),np.cos(param.theta),
                    param.Na,I1Tvol,aTground)
                    
    print 'a: {0} avrai: {1} bvrai: {2}'.format(a,avrai,bvrai)    
    print a.shape,avrai.shape
    print 'Tground_vrai'
    bl.printm(param.T_ground)
    print 'aTground/a_vrai'
    bl.printm(aTground/param.get_a())
    print 'Tvol_vrai'
    bl.printm(param.T_vol)
    print 'I1Tvol/I1_vrai'
    bl.printm(I1Tvol/param.get_I1())
    print 'Ups_vraie'
    bl.printm(param.get_upsilon_gt())    
    print 'Ups_reonstruit'
    bl.printm(Ups_reconstr)
    print '||Ups-Ups_recon||={0}'.format(npl.norm(param.get_upsilon_gt()-Ups_reconstr))
    
    args=[param.get_upsilon_gt(),\
          param.get_kzlist(),np.cos(param.theta),param.Na,\
          I1Tvol,aTground,param.N,param.get_gtlist()]
          
    V=e.mlogV((param.h_v,param.sigma_v),*args)
    print 'Vraissemb={0}'.format(V)
    Vvrai=np.log(np.pi**(3*param.Na)*npl.det(param.get_upsilon_gt()))+3*param.N*param.Na
    print 'Vraissemb si Ups_est=Ups_vrai et hv,sig=hvvrai,sigvrai: {0}'.format(Vvrai)

def test_imshow_mlogV():
    """Calcul du critère de logvraissemblance et d'EQM"""
    np.set_printoptions(precision=3)
    param = lp.load_param('DB_1')
    param.N=1#cas non bruité
    W = tom.UPS_to_MPMB(param.get_upsilon_gt())    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,param.Na)

    R_t,C_t,_ = tom.sm_separation(W_norm,Np=3,Na=param.Na)        
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,param.Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    #choix du a et b
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))][0]
    
    bvrai = tom.b_true2(R_t,param)
    avrai = tom.a_true(R_t,param)
    
    
    
    _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,avrai,bvrai)
    aTground,I1Tvol=tom.denormalisation_teb(W,param.Na,Cg,Cv)
    
    arg=[param.get_upsilon_gt(),\
          param.get_kzlist(),np.cos(param.theta),param.Na,\
          I1Tvol,aTground,param.N,param.get_gtlist()]
          
    Ups_reconstr=e.Ups_reconstr2(param.h_v,param.sigma_v,param.get_gtlist(),
                    param.get_kzlist(),np.cos(param.theta),
                    param.Na,I1Tvol,aTground)
    
    Npts=50
    mat_CritV=np.zeros((Npts,Npts))               
    mat_EQM=np.zeros((Npts,Npts))
    sigmin=0.001
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(param.get_kzlist())
    
    Ups_est=tom.UPS_to_MPMB(W)
    X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
    xoptV=opti.minimize(e.mlogV,X_0,\
                        (param.get_upsilon_gt(),param.get_kzlist(),\
                        np.cos(param.theta),param.Na,I1Tvol,aTground,\
                        param.N,param.get_gtlist()),\
                        method ='TNC',bounds=[(hvmin,hvmax),(sigmin,sigmax)],\
                        options={'ftol':10**-100})
    (hvopt,sigvopt)=xoptV.get('x')    
    
    
    
    vec_sigv=np.linspace(sigmin,sigmax,Npts)
    vec_hv=np.linspace(hvmin,hvmax,Npts)    
    idxhvvrai=np.argmin(np.abs(vec_hv-param.h_v))
    idxhvopt=np.argmin(np.abs(vec_hv-hvopt))
    
    idxsigvvrai= np.argmin(np.abs(vec_sigv-param.sigma_v))   
    idxsigvopt= np.argmin(np.abs(vec_sigv-sigvopt))   
    for ihv,hv in enumerate(vec_hv):
        for isig,sigv in enumerate(vec_sigv):
            mat_CritV[ihv,isig] = e.mlogV((hv,sigv),*arg)
            Ups_r=e.Ups_reconstr2(hv,sigv,param.get_gtlist(),
                    param.get_kzlist(),np.cos(param.theta),
                    param.Na,I1Tvol,aTground)
            mat_EQM[ihv,isig] = npl.norm(Ups_r-param.get_upsilon_gt())
    idxminhvbrut,idxminsigbrut = np.where(mat_CritV==np.min(mat_CritV))


    plt.imshow(mat_CritV,origin='lower',
               vmin=np.min(mat_CritV),vmax=np.min(mat_CritV)*1.10)
    plt.hold(True)
    plt.plot(idxsigvvrai,idxhvvrai,'or',markersize=10,label='vrai')    
    plt.plot(idxsigvopt,idxhvopt,'sg',markersize=6,alpha=0.5,label='opt')
    plt.hold(False)
    plt.axis('tight')          
               
    plt.figure()
    plt.imshow(mat_EQM,vmin=0,vmax=10,origin='lower')
    plt.hold(True)
    plt.plot(idxsigvvrai,idxhvvrai,'or',markersize=5,label='vrai')    
    plt.plot(idxsigvopt,idxhvopt,'sg',markersize=6,alpha=0.5,label='opt')
    plt.hold(False)
    plt.axis('tight')
    
    
    """
    plt.figure()    
    bl.iimshow(mat_CritV[30:,:20],plot_contour='minmax',show_min='')
    plt.hold(True)
    plt.plot(idxsigvvrai,idxhvvrai,'or',markersize=5,label='vrai')
    plt.plot(idxminsigbrut,idxminhvbrut,'sc',markersize=4,label='min_brut')
    plt.plot(idxsigvopt,idxhvopt,'sg',markersize=6,alpha=0.5,label='opt')
    plt.hold(False)
    """
#    plt.legend()
#    plt.colorbar()
    
def test_echantillon_sigma():    
    """Observons la variation de gamma_v en fonction
    du type d'echantillonage en sig_v '(tt étant fixé par ailleurs)"""
    sig_min=0.001
    sig_max=1
    nbsig=20
    vec_sig_log=np.logspace(np.log10(sig_min),np.log10(sig_max),nbsig)
    vec_sig_lin = np.linspace(sig_min,sig_max,nbsig)    
    gv_lin = np.zeros(nbsig,dtype='complex')
    gv_log = np.zeros(nbsig,dtype='complex')
    
    hv=50
    costheta=np.cos(np.pi/4)
    kz=0.08
    for i in range(vec_sig_lin.size):
        gv_lin[i]=e.gammav(hv,vec_sig_lin[i],costheta,kz)
        gv_log[i]=e.gammav(hv,vec_sig_log[i],costheta,kz)
    
    plt.close('all')
    plt.figure()
    plt.polar(np.angle(gv_lin),np.abs(gv_lin),'s-b',label='echant lineaire')
    plt.polar(np.angle(gv_log),np.abs(gv_log),'*-r',label='echant log')
    plt.legend()
    
    plt.figure()
    plt.plot(np.abs(np.diff(gv_log)),'*-r',label='diff echant log')
    plt.plot(np.abs(np.diff(gv_lin)),'*-b',label='diff echant lineaire')
    plt.legend()
    
def test_echantillon_hv():    
    print 'Pensez à ranger test_echant_hv'
    kz=0.7    
    hv=30
    costheta=np.cos(np.pi/4)
    sigma_v=0.0345
     
    hv_min=8
    hv_max=2*np.pi/kz    
    nbhv=20
    vec_hv_log=np.logspace(np.log10(hv_min),np.log10(hv_max),nbhv)
    vec_hv_lin = np.linspace(hv_min,hv_max,nbhv)
    gv_lin = np.zeros(nbhv,dtype='complex')
    gv_log = np.zeros(nbhv,dtype='complex')
    
    for i in range(vec_hv_lin.size):
        gv_lin[i]=e.gammav(vec_hv_lin[i],sigma_v,costheta,kz)
        gv_log[i]=e.gammav(vec_hv_log[i],sigma_v,costheta,kz)
    
    plt.close('all')
    plt.figure()
    plt.polar(np.angle(gv_lin),np.abs(gv_lin),'s-b',label='echant lineaire')
    plt.polar(np.angle(gv_log),np.abs(gv_log),'*-r',label='echant log')
    plt.legend()
    
    plt.figure()
    plt.plot(np.abs(np.diff(gv_log)),'*-r',label='diff echant log')
    plt.plot(np.abs(np.diff(gv_lin)),'*-b',label='diff echant lineaire')
    plt.legend()

def test_Rv_from_gamma_et_recip():
    
    vec_gm = np.random.randint(1,7,size=3)+1j*np.random.randint(1,7,size=3)
    Rtv=tom.Rv_from_gamma(vec_gm)
    print vec_gm
    print Rtv
    vec2=tom.gamma_from_Rv(Rtv)
    print 'vec_gm init {0} vec_bi_trans {1}'.format(vec_gm,vec2)
    pdb.set_trace()

def ajout_bcr_simu_montecarl():
    """script pour ajouter une bcr aux serie de test d'estimation SKP
    (fonction monte_carl_estim_ecart_ang_tot)"""
    
    dirname='D:\PIERRE CAPDESSUS\Python\Code_Pierre\monte_carl_estim_ecar_ang_opt\Test8' 
    ddata=bl.load_npy_data(dirname)
    A=0.95
    E=200
    X=0.2
    vec_N=ddata['vec_N']
    P=ddata['P']
    scenario='sigv_inconnu'    
    param=mb.param_rvog()
    param.T_vol,param.T_ground=mb.Tv_Tg_from_A_E_X(A,E,X)
    param.k_z=[0.1,0.15]
    param.Na=len(param.k_z)+1
    param.sigma_v=0.0345
    param.h_v=10
    param.theta=45*np.pi/180
    param.z_g=0
    param.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    Ystd=np.sqrt(ddata['varhvJ'])
    interv_std=1/np.sqrt(2*P-1)*Ystd*2
    
    vec_bcr_hv = mb.bcr_mb_N_influence(param,A,E,X,vec_N,scenario)
    pt.plot_err(vec_N,Ystd,interv_std,Y_vrai=vec_bcr_hv,
                xscale='log',yscale='log')
    plt.ylim((10**-2,10))
    filename=dirname+'\\stdhv_bcrhv.png'
    bl.save_fig(filename)

def test_get_Fisher():
    A=0.95
    E=200
    X=0.2
    Tvol,Tground=mb.Tv_Tg_from_A_E_X(A,E,X)
    gamma_t_mat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    p=mb.param_rvog(N=10000,k_z=[0.1,0.15],theta=45*np.pi/180,T_vol=Tvol,
                    T_ground=Tground,h_v=30,z_g=0,sigma_v=0.0345,
                    mat_gamma_t=gamma_t_mat)
    F=p.get_fisher()
    invF=npl.inv(F)
    bcrhv=invF[19,19]
    print np.sqrt(bcrhv)
    pdb.set_trace()
    
def test_retranch_phig_W():
    """Test du retranchage de la pahse du sol.
    Comparaison des region de cohérences avec et sans phase du sol"""
    param = lp.load_param('DB_1')
    param.z_g = 40
    param.N=100
    data = tom.TomoSARDataSet_synth(param)
    W = data.get_W_k_rect(param,param.N)
    #W = tom.UPS_to_MPMB(param.get_upsilon_gt())
    W_rot = tom.retranch_phig_W(param.z_g,W,param.get_kzlist())
    pt.plot_rc_mb(tom.MPMB_to_UPS(W))
    pt.plot_rc_mb(tom.MPMB_to_UPS(W_rot))
    
if __name__ == "__main__":    
    #test_estim_ecart_ang_opt_2()
    #test_selection_ground_selection_MB()  
    #test_montecarl_estim_ecart_ang_opt()
    #test_montecarl_estim_ecart_ang_opt2()
    #test_denormalisation_tebaldini()
    #test_selection_ground_selection_MB()
    #test_estim_ecart_ang()
    #test_estim_ecart_ang()
    #test_estim_ecart_ang_tot()
    #test_Rv_from_gamma_et_recip()
    #test_estim_ecart_ang_scal()
    #test_estim_ecart_ang()
    #test_selection_ground_selection_MB()
    #test_estim_ecart_ang_pol()
    #test_estim_ecart_ang_scal()
    #test_monte_carlo_estim_ang_scal()
    #test_plot_cu_plus_tebaldini_plus_legend()
    #test_get_Fisher()   
    #test_estim_V_scal()
    #test_estim_Tvol_Tground_from_W()
    #test_imshow_mlogV()
    #test_V_scal()
    #test_V_scal_gt_inconnu()
    #test_estim_V_scal()
    #test_estim_V_scal_gt_inconnu()
    #test_retranch_phig_W()
    #estim_mlogV_gt_inconnu_all()
    #test_mlogV_gt_inconnu()
    #test_mlogV_gt_inconnu()
    #test_mlogV_Tg_Tv_known()
    #test_estim_mlogV_Tg_Tv_known()
    test_estim_ecart_ang_scal()