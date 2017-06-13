# -*- coding: utf-8 -*-
from __future__ import division
import time
import plot_tomo as pt
import tomosar_synth as tom
import sys
import numpy as np
import numpy.linalg as npl
import matplotlib as mpl
import matplotlib.pyplot as plt
import RVoG_MB as mb
import basic_lib as bl
import optimisation as opt
import scipy.optimize as opti
import pdb
import load_param as lp
plt.ion()   

def I1(hv,sigmav,costheta):    
    """`I1 = 1-a /alpha`"""
    
    alpha = 2*sigmav/costheta
    #limite/Prolongement par continuite en 0
    epsi = np.finfo(float).eps
    if hv<epsi or sigmav<epsi:
        return hv
    elif costheta<epsi:
        return 0
    else:        
        I1 = (1-np.exp(-alpha*hv))/alpha        
        if np.isnan(I1) or np.isinf(I1): pdb.set_trace()
        return I1
    
def I2(hv,sigmav,costheta,kz):
    alpha = 2*sigmav/costheta
    return (np.exp(1j*kz*hv)-np.exp(-alpha*hv))/(1j*kz+alpha)     
    
def gammav(hv,sigmav,costheta,kz):
    gv=0+1j*0
    if type(hv)==complex: print 'dans gmmav hv cpx !';pdb.set_trace()
    if np.isnan(float(hv)) or np.isnan(float(sigmav)):
        return float(np.nan)
    else:
        if I1(hv,sigmav,costheta) != 0 :
            gv=I2(hv,sigmav,costheta,kz)/I1(hv,sigmav,costheta)
        else:
            gv=np.inf
        return gv
    
def gammavgt(hv,sig,costheta,kz,gt):
    gvgt=0.+0*1j
    gvgt=gammav(hv,sig,costheta,kz)*gt
    return gvgt 
    
def J(X,*arg):
    """Calcul du critère J en fonction des 2 var :  hv, sig
    
    **Entrées**    
        * *X* : variable du critère X = (hv,sigma_v)        
        * *arg* : tuple. Paramètres necessaire au calcul de J
            * *arg[0]* : mat R_t (decomposition SKP)
            * *arg[1]* : vec_kz, ensemble des baselines dans l'ordre {kzij , i<j}
            * *arg[2]* : costheta : cos(angle incidence)
            
    **Sortie**
        * *crit* : critère J
    """
            
    vec_gm=arg[0] 
    vec_kz=arg[1]
    costheta=arg[2]
    
    if vec_gm==None or costheta==None or vec_kz==None:
        print '=== ! Erreur Manque des paramètrees! ==='
        return np.nan
    else:      
        hv=X[0]
        sigv=X[1]
        Nb_gma = vec_gm.size
        crit=0
        vec_gtest=np.zeros(Nb_gma)
        for i in range(Nb_gma):
            gvi = gammav(hv,sigv,costheta,vec_kz[i])
            crit = crit + np.abs(vec_gm[i]- gvi/np.abs(gvi)*np.abs(vec_gm[i]))**2
            vec_gtest[i]=vec_gm[i]/gvi
            
        if np.sum(vec_gtest>1)>=1:
            return np.inf
                        
        return crit
        
def J2(X,*arg):
    """Calcul du critère J2 en fonction des 2 var :  hv, sig
    
    **Entrées**
	* *X* : variable du critere. X = (hv,sigma_v)
	* *arg* : paramètres necessaire au calcul de J2
		* *arg[0]* : mat R_t contient les gamma_mesuré(decomposition SKP)
		* *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}
		* *arg[2]* : costheta : cos (angle incidence)
        
    **Sortie**
        * *crit* : critère J2

    """
        
    vec_gm=arg[0]
    vec_kz=arg[1]
    costheta=arg[2]
    
    if vec_gm==None or costheta==None or vec_kz==None:
        print '=== ! Erreur Manque des paramètrees d\entré bro ! ==='
        return np.nan
    else:      
        hv=X[0]
        sigv=X[1]
        Nb_gma = vec_gm.size
        crit=0
        vec_gtest=np.zeros(Nb_gma)
        for i in range(Nb_gma):
            gvi = gammav(hv,sigv,costheta,vec_kz[i])
            crit = crit + np.abs(vec_gm[i]/np.abs(vec_gm[i])- gvi/np.abs(gvi))**2
            vec_gtest[i]=vec_gm[i]/gvi
            
        if np.sum(vec_gtest>1)>=1:
            return np.inf
            
        return crit
        
def J_tot(X,*arg):
    """Calcul du critère J en fonction des 3 var : b, hv, sig
        
    **Entrées**
        * *X* : Variable du critère X = (b,hv,sigma_v)
        * *arg* : tuple. paramètres necessaires au calcul de J_tot
            * *arg[0]* : mat R_t (decomposition SKP)
            * *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}
            * *arg[2]* : costheta : cos (angle incidence)
    **Sortie**
        * *crit* : critère J_tot    """
    
    R_t = arg[0]
    vec_kz=arg[1]
    costheta=arg[2]
    
    if len(arg)==0:
        print '(J_tot): pas d argument!'
        return np.nan
    else:
        b=X[0]
        hv=X[1]
        sig=X[2]
        Rv=b*R_t[0]+(1-b)*R_t[1]
        vec_gm=tom.gamma_from_Rv(Rv)        
        Y=np.array([hv,sig])        
        J_tot = J(Y,vec_gm,vec_kz,costheta)
    return J_tot
    
def J_scal(b,*arg):        
    """Calcul du critère J vu comme fonction (scalaire) de b (decomposition SKP),
    
    Pour un *b* donnée, on mimnimise le critère J par rapport à (hv,sigv)
    
    **Entrées**    
        * *b* : variable du critère (b de la decomposition SKP)
        * *arg* : tuple. Paramètres necessaire au calcul de J_tot
            * *arg[0]* : mat R_t (decomposition SKP)
            * *arg[1]* : vec_kz, ensemble des baselines dans l'ordre {kzij , i<j}
            * *arg[2]* : costheta : cos(angle incidence)
            * *arg[3]* : U0 : initial guess (hv0,sigv0)
            
    **Sortie**
        * *J_scal* : critère J_scal
    """
    
    R_t = arg[0]
    vec_kz = arg[1]
    costheta = arg[2]
    U0 = arg[3]
    
    if len(arg)==0:
        print '(J_tot): pas d argument!'
        return np.nan
    else:        
        sigmin=0.01
        sigmax=0.1
        hvmin=5.
        hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
        
        Rv=b*R_t[0]+(1-b)*R_t[1]
        vec_gm=tom.gamma_from_Rv(Rv)
        X_0 = U0
        xoptJ=opti.minimize(fun=J,x0=X_0,
                            args=(vec_gm,vec_kz,costheta),method ='TNC',
                            bounds=[(hvmin,hvmax),(sigmin,sigmax)],
                            options={'ftol':10**-8})                            
        J_scal=xoptJ.get('fun')
        return J_scal

def J2_scal(b,*arg):        
    """Calcul du critère J2 vu comme fonction (scalaire) de b.
    
    **Entrée**
        * *b* : variable de départ (b de la decomposition SKP)
        * *arg* : paramètres necessaire au calcul de J2_scal
            * *arg[0]* : mat R_t (decomposition SKP)
            * *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}
            * *arg[2]* : costheta : cos (angle incidence)
            * *arg[3]* : Initial guess pour la minisation de J2
    
    **Sortie**
        * *J_scal* : critère J_scal
    """
    
    R_t = arg[0]
    vec_kz=arg[1]
    costheta=arg[2]
    U0 = arg[3]    
    
    if len(arg)==0:
        print '(J_tot): pas d argument!'
        return np.nan
    else:        
        sigmin=0.01
        sigmax=0.1
        hvmin=5.
        hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité        
        Rv=b*R_t[0]+(1-b)*R_t[1]
        vec_gm=tom.gamma_from_Rv(Rv)
        #X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
        X_0 = U0        
        xoptJ=opti.minimize(fun=J2,x0=X_0,
                            args=(vec_gm,vec_kz,costheta),method ='TNC',
                            bounds=[(hvmin,hvmax),(sigmin,sigmax)],
                            options={'ftol':10**-8})                            
        J_scal=xoptJ.get('fun')
        return J_scal

def V_scal(b,*arg):        
    """Calcul du critère de vraissemblance vu comme fonction (scalaire) de b.
    
    Pour un *b* donnée, on mimnimise le critère de vraissemblance par rapport à (hv,sigv)
    
    **Entrées**
        * *b* : variable de départ (b de la decomposition SKP)
        * *arg* : paramètres necessaire au calcul de V_scal
            * *arg[0]* : W matrice de covar estimé (base MPMB)
            * *arg[1]* : mat R_t (decomposition SKP)
            * *arg[2]* : mat C_t (decomposition SKP)
            * *arg[3]* : vec_kz, ensemble des baseline dans l'ordre {kzij,i<j}
            * *arg[4]* : costheta : cos (angle incidence)
            * *arg[5]* : a : paramètre de la décomposition SKP
            * *arg[6]* : Na : Nombre d'antennes
            * *arg[7]* : N : Taille d'échantillon
            * *arg[8]*: vec_gt : vecteur contenant les décohérence temp dans l'ordre {gt_ij | i<j}
            * *arg[9]* : zg: altitude sol
            
    **Sortie**
        * *V_scal* : critère
    """
    
    W = arg[0]
    R_t = arg[1]
    C_t = arg[2]
    vec_kz = arg[3]
    costheta = arg[4]
    a = arg[5]
    Na = arg[6]
    N = arg[7]
    vec_gt = arg[8]
    zg = arg[9]
    
    if len(arg)==0:
        print '(J_tot): pas d argument!'
        return np.nan
    else:
        sigmin=0.01
        sigmax=0.1
        hvmin=5.
        #le min des hauteurs d'ambiguité
        hvmax=2.*np.pi/np.max(vec_kz)
        
        _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,b)
        aTground,I1Tvol=tom.denormalisation_teb(W,Na,Cg,Cv)        
        Ups_est=tom.MPMB_to_UPS(W)
        X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
        xoptV=opti.minimize(fun=mlogV,x0=X_0,\
                            args=(Ups_est,vec_kz,costheta,Na,I1Tvol,aTground,N,vec_gt,zg),\
                            method ='TNC',bounds=[(hvmin,hvmax),(sigmin,sigmax)],\
                            options={'ftol':10**-10})
        V_scal=xoptV.get('fun')        
        return V_scal

def V_scal_gt_inconnu(b,*arg):        
    """Calcul du critère de vraissemblance vu comme fonction (scalaire) de b.
    avec gt.
    
    Pour un *b* donnée, on mimnimise le critère de vraissemblance par rapport à (hv,sigv)\n
    **NB**: contrairement à :func:`~estimation.V_scal` les gt sont supposés inconnus.
    
    **Entrées**
        * *b* : variable de départ (b de la decomposition SKP)
        * *arg* : paramètres necessaire au calcul de V_scal
            * *arg[0]* : W matrice de covar estimé (base MPMB)
            * *arg[1]* : mat R_t (decomposition SKP)
            * *arg[2]* : mat C_t (decomposition SKP)
            * *arg[3]* : vec_kz, ensemble des baseline dans l'ordre {kzij,i<j}
            * *arg[4]* : costheta : cos (angle incidence)
            * *arg[5]* : a : paramètre de la décomposition SKP
            * *arg[6]* : Na : Nombre d'antennes
            * *arg[7]* : N : Taille d'échantillon
            * *arg[8]* : zg : altitude du sol
    
    **Sorties**
        * *V_scal* : critère """

    W = arg[0]
    R_t = arg[1]
    C_t = arg[2]
    vec_kz = arg[3]
    costheta = arg[4]
    a = arg[5]
    Na = arg[6]
    N = arg[7]
    zg = arg[8]
    print '-> b : {0}'.format(b)
    if len(arg)==0:
        print '(V_scal_gt_inconnu): pas d argument!'
        return np.nan
    else:
        sigmin=0.01
        sigmax=0.1
        hvmin=5.
        hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
        
        Ngt=mb.get_Nb_from_Na(Na)
        _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,b)
        aTground,I1Tvol=tom.denormalisation_teb(W,Na,Cg,Cv)
        Ups_est=tom.MPMB_to_UPS(W)
        #ex en DualB: X_0=(hv_0,sig_0,gt12_0,gt13_0,gt23_0)
        X_0 = np.hstack((np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2]),np.ones((Ngt))))
        
        xoptV=opti.minimize(fun=mlogV_gt_inconnu_all,x0=X_0,\
                            args=(Ups_est,vec_kz,costheta,Na,I1Tvol,aTground,N,zg),\
                            method ='TNC',
                            bounds=[(hvmin,hvmax),(sigmin,sigmax)]+[(0,1)]*Ngt,
                            options={'ftol':10**-10})
        V_scal=xoptV.get('fun')        
        return V_scal

def estim_rvog_crit_divers(W_norm,param):
    """Estimation des paramètres RVoG hv,sigv et gammat à partir de la méthode SKP.
    
    Renvoie les couples (hv,sigv) sur chaque baseline. L'intersection 
    des courbes dans le plan donnent le couple (hv,sigv) vrai.
    
    Choix des valeurs pour a et b:\n
    Pour a : valeur qui maximise le coherence du sol\n
    Pour b : valeur maximisant la diversité des matrice de structures\n
        \t(cf Algebrical synthetiseic of forest scenarios. Tebaldini).
    
    **Entrées**
        * *W_norm* : matrice de covariance (base MPMB) normalisée (PS+Tebaldini)
        * *param* : classe de paramètres RVoG
    
    """
        
    Np=3
    Na=int(np.floor(W_norm.shape[0]/Np))
    
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    
    #Selection du a maximaisant la coherence du sol sur la baseline minimale
    #si kzij = delta_kz*(j-i) => on regarde sur kz12
    #On suppose que R_t[0]=Rg est le sol
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]
    
    Nb_zoom = 3
    Nb_pts = 50
    bmin = interv_b[0][0]
    bmax = interv_b[0][1]
    print '=== init : bmin {0} bmax {1} ===='.format(bmin,bmax)
    J = np.zeros((Nb_zoom,Nb_pts))
    for i in range(Nb_zoom):                
        b_test = np.linspace(bmin,bmax,Nb_pts)
        print '==== bmin {0} bmax {1} ===='.format(bmin,bmax)       
        for idx_b,b in enumerate(b_test):
            _,R1,R2,_,_ = tom.value_R_C(R_t,C_t,a,b)
            J[i,idx_b]=tom.criterion(R1,R2)           
        idx_maxJ = np.argmax(J[i,:])
        bmin = b_test[idx_maxJ]
        bmax = b_test[np.min([idx_maxJ,b_test.size])]
    b=(bmax+bmin)/2
    
    _,Rg,Rv,_,_ = tom.value_R_C(R_t,C_t,a,b)
    
    #gamma_v12 = Rv[0,1]
    gamma_v12=param_MB.get_gamma_v(0,1)*param_MB.get_gamma_t(0,1) #VRAI
    #gamma_v13 = Rv[0,2]
    gamma_v13 = param_MB.get_gamma_v(0,2)*param_MB.get_gamma_t(0,2)#VRAI
    #Obtention des paramètres hv,sigmav et gammat
    ima_nv,ima_phiv =load_nv_phiv()
    hv_list1,sig_list1,gamm_list1=db.from_gama_to_hv_sigma_gamat(
                                            ima_phiv[:],ima_nv[:],\
                                            np.cos(param.theta),\
                                            param.k_z[0],gamma_v12)        
    
    hv_list2,sig_list2,gamm_list2=db.from_gama_to_hv_sigma_gamat(
                                            ima_phiv[:],ima_nv[:],\
                                            np.cos(param.theta),\
                                            param.k_z[1],gamma_v13)        
    #print hv_list1,sig_list1
    #print hv_list2,sig_list2
    
    
    plt.figure()
    plt.plot(sig_list1,hv_list1,'-*b',sig_list2,hv_list2,'-*g')
    
    return sig_list1,hv_list1,gamm_list2,sig_list2,hv_list2,gamm_list2
    
def estim_rvog_b_free(W_norm,param):
    """Estimation des paramètres RVoG à partir 
    de la matrice de covariance (base MPMB) normalisé (PS+Tebaldini)
    basée sur la méthode SKP.\n
    Il faut choisir des valeurs pour a et b.\n
    Pour a : valeur qui maximise le coherence du sol\n
    Pour b : pour chaque b on determine le couple hv_sigma_v
    
    **Entrées**
        * *W_norm* : matrice de covariance (base MPMB) normalisée (PS+Tebaldini)
        * *param* : classe de paramètres RVoG"""   
    
    nb_pts = 2
    hv = np.zeros((nb_pts,1))    
    sigma=np.zeros((nb_pts,1))    
    gt1=np.zeros((nb_pts,1))    
    gt2=np.zeros((nb_pts,1))
    
    ima_nv,ima_phiv =load_nv_phiv('/data2/pascale/')
    
    Np=3
    Na=int(np.floor(W_norm.shape[0]/Np))
    
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    #choix du a
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]
    
    _,Rg,_,_,_ = tom.value_R_C(R_t,C_t,a,0)
    phi_g12 = np.angle(Rg[0,1])
    phi_g13 = np.angle(Rg[0,2])
    
    #choix du b
    b_test=np.linspace(interv_b[0][0],interv_b[0][1],nb_pts)
    gamma12 = R_t[1][0,1]
    gamma13 = R_t[1][0,2]
    for idx_b,b in enumerate(b_test):
          _,_,Rv,_,_ = tom.value_R_C(R_t,C_t,a,b)
          gamma12 = Rv[0,1]*np.exp(-1j*phi_g12)
          gamma13 = Rv[0,2]*np.exp(-1j*phi_g13)
          
          hv[idx_b],sigma[idx_b],gt1[idx_b],gt2[idx_b]=\
              db.from_gama1_gama2_to_hv_sigma_gamat1_gamat2(\
              ima_phiv[:],ima_nv[:],np.cos(param.theta),\
              param.k_z[0],param.k_z[1],gamma12,gamma13)
                    
          if np.isnan(hv[idx_b]) : print 'hv = Nan!' 

    return hv,sigma,gt1,gt2
       
def estim_dpct_kz(W_norm,param):    
    """Estimation des paramètres RVog par projection des gammav 
    sur une même baseline
    
    **Entrées**:
        * *W_norm* : Matrice de covariance normalisé (PS+Tebald) base MPMB
        * *param*  : classe de paramètre RVoG
             
    **Sorties**
        * *mat_S* : contient la sum des diff possible du vecteur des gammav projetés sur une mm baseline. ceci pour chaque b
                    mat_S.shape=(nb_gt**Na,nb_ptb)
        * *idx_minS* : idx du min de mat_S[:,idxb] -> idx_min est un vec taille nb_ptb
        * *minS* : valeur du min de mat_S[:,idxb] -> minS est un vec taille nb_ptb
        * *chx_min* : vcteur de choix correspondant à idx_minS
        * *mat_gt_opt,hv_opt,sig_opt*: paramètres correspondant à la minimisation (paramètres estimés) vec dependant de b
        * *b_min,hv_min,sig_min,gt_min*: minimisation par rapport à b des paramètres précédents.
        * *mat_gamgt,mat_gamgt_proj* : matrice des cohérences, matrices des cohérences projeté sur lamême baseline    
        
    """
    start = time.time()
    Np=3
    Na=int(np.floor(W_norm.shape[0]/Np))
    
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b=tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    nb_ptb=10
    #choix du a
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]
    b_test=np.linspace(interv_b[0][0],interv_b[0][1],nb_ptb)

    nb_gt=20
    vec_gamgt = np.zeros((1,nb_gt),dtype='complex')    
    vec_gamgt_proj = np.zeros((1,nb_gt),dtype='complex')
    mat_S = np.zeros((nb_gt**Na,nb_ptb))    
    minS = np.zeros(nb_ptb)
    idx_minS= np.zeros(nb_ptb)
    chx_min =np.zeros((Na,nb_ptb))    
    vec_gt_chx_min = np.zeros((Na,nb_ptb))
    mat_gamgt =np.zeros((nb_ptb,Na,nb_gt),dtype='complex')
    mat_gt_opt = np.zeros((Na,nb_ptb))
    mat_gvgt_opt = np.zeros((Na,nb_ptb),dtype='complex')
    mat_gv_opt = np.zeros((Na,nb_ptb),dtype='complex')
    mat_gt=np.zeros((nb_ptb,Na,nb_gt),dtype='complex')
    mat_gamgt_proj =np.zeros((nb_ptb,Na,nb_gt),dtype='complex')
    hv_opt = np.zeros((Na,nb_ptb))
    hv_min =np.zeros(Na)
    sig_opt = np.zeros((Na,nb_ptb))
    sig_min = np.zeros(Na)
    ima_nv,ima_phiv =load_nv_phiv('/data2/pascale/')
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    
    for idx_b,b in enumerate(b_test):
        print '-----------'
        print 'b',b
        _,_,Rv,_,_ = tom.value_R_C(R_t,C_t,a,b)           
        gamma = tom.gamma_from_Rv(Rv)
        for i in range(Na):
            vec_gt = np.linspace(1,0.98/(np.abs(gamma[i])),nb_gt)
            vec_gamgt = gamma[i]*vec_gt
            mat_gamgt[idx_b,i,:]=vec_gamgt
            mat_gt[idx_b,i,:]=vec_gt
            for j in range(nb_gt):         
                
                hv,sigma=db.from_gama_to_hv_sigma(ima_phiv,ima_nv,costheta,\
                                               vec_kz[i],vec_gamgt[j])
                                
                #projection sur labaseline de ref : kz12
                if np.isnan(gammav(hv,sigma,costheta,vec_kz[0]))==False:                    
                    vec_gamgt_proj[0,j] = gammav(hv,sigma,costheta,vec_kz[0])
                else:
                    vec_gamgt_proj[0,j] = 4
                    
            mat_gamgt_proj[idx_b,i,:]=vec_gamgt_proj
            
        mat_S[:,idx_b],idx_minS[idx_b],minS[idx_b]\
                       ,chx_min[:,idx_b] = opt.min_sum_diff(mat_gamgt_proj[idx_b,:,:])
                       
        mat_gt_opt[:,idx_b]= np.array([mat_gt[idx_b,j,chx_min[j,idx_b]] for j in range(Na)])
        mat_gvgt_opt[:,idx_b]= np.array([mat_gamgt[idx_b,j,chx_min[j,idx_b]] for j in range(Na)])
        mat_gv_opt[:,idx_b]=mat_gvgt_opt[:,idx_b]/mat_gt_opt[:,idx_b]
        
        for j in range(Na):            
            hv_opt[j,idx_b],sig_opt[j,idx_b]=db.from_gama_to_hv_sigma(\
                                                    ima_phiv,ima_nv,costheta,\
                                                    vec_kz[j],mat_gvgt_opt[j,idx_b])
    b_min = np.argmin(minS)
    hv_min = hv_opt[:,b_min]
    sig_min = sig_opt[:,b_min]    
    gt_min = mat_gt_opt[:,b_min]                                       
    print 't_exec estim_dpct_kz : {0}'.format(time.time()-start)
    return mat_S,idx_minS,minS,chx_min,mat_gt_opt,hv_opt,sig_opt,\
           mat_gamgt,mat_gamgt_proj,b_min,hv_min,sig_min,gt_min
           
def estim_ecart_ang(W,param):
    """Estimation des paramètres RVog par miminsation de critères.
    
    Plusieurs critères sont minimisés : J,J2,l'EQM et la Vraissemb. Attention, CODE pour du dual_baseline.
    
    **Entrée**   
        * *W* : matrice de covariance  (base MPMB)
        * *param* : class de paramètres RVoG
        
    **Sortie** 
        * *J,J2,Ressemb,ErrQM* : valeurs des critères en fction de (nb_ptb,nb_hv,nb_sig)
        * *vec_hv,vec_sig,vec_b* : grille de maillage pour la minimisation
        * *hv_J,hv_J2,hv_MV,hv_EQM* : valeurs des hv estimés en fction de b
        * *sig_J,sig_J2,sig_MV,sig_EQM* : valeurs des sig estimés en fction de b
        * *vec_gt_J,vec_gt_J2,vec_gt_MV,vec_gt_EQM* :valeur des gt estimés en fction de b"""
        
    start = time.time()
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    nb_ptb= 500
    nb_hv = 50
    nb_sig = 50
    vec_hvmin = np.zeros(Na)    
    vec_sigmin=np.zeros(Na)    
    vec_hvmax = np.zeros(Na)    
    vec_sigmax=np.zeros(Na)    
    
    hv_J=np.zeros(nb_ptb)
    hv_J2=np.zeros(nb_ptb)
    hv_MV = np.zeros(nb_ptb)
    hv_EQM=np.zeros(nb_ptb)
    
    sig_J=np.zeros(nb_ptb)
    sig_J2=np.zeros(nb_ptb)
    sig_MV=np.zeros(nb_ptb)
    sig_EQM=np.zeros(nb_ptb)    
    
    vec_gt_J=np.zeros((nb_ptb,Nbase))
    vec_gt_J2=np.zeros((nb_ptb,Nbase))
    vec_gt_MV=np.zeros((nb_ptb,Nbase))
    vec_gt_EQM=np.zeros((nb_ptb,Nbase))
    
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    Ups = tom.MPMB_to_UPS(W)
    Ups_norm = tom.MPMB_to_UPS(W_norm)
    Ups_est = tom.deblanch(Ups_norm,E)
    
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b=tom.ground_selection_MB(R_t,interv_a,interv_b)        
    
    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    

    #test sur b    
    vec_b=np.linspace(interv_b[0][0],interv_b[0][1],nb_ptb)
    b_vrai=tom.b_true(R_t,param)
    
    #critères J et J2
    J = np.zeros((nb_ptb,nb_hv,nb_sig),dtype='double')
    J2 = np.zeros((nb_ptb,nb_hv,nb_sig),dtype='double')
    
    #estimation de la vrassemblance sur l'ensemble des hv sigv
    Ressemb = np.zeros((nb_ptb,nb_hv,nb_sig),dtype='double')
    ErrQM= np.zeros((nb_ptb,nb_hv,nb_sig),dtype='double')

    vec_sig=np.logspace(np.log10(0.01),np.log10(0.1),nb_sig) #TEST
    hvmin_dur=5.
    hvmax_dur=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
    vec_hv=np.linspace(hvmin_dur,hvmax_dur,nb_hv)   
    
    print 'hv_test: [{0}  {1}]'.format(vec_hv[0],vec_hv[-1])
    print 'sig_test: [{0}  {1}]'.format(vec_sig[0],vec_sig[-1])

    ihvvrai = np.argmin(np.abs(vec_hv-param.h_v)) 
    idxb_numvrai =np.argmin(np.abs(vec_b-tom.b_true(R_t,param)))
    jsigvrai = np.argmin(np.abs(vec_sig-param.sigma_v))
    
    #calcul des critères Ji et de la vraissemblance
    for idxb,b in enumerate(vec_b):
        
        _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,b)
        vec_gm=tom.gamma_from_Rv(Rv)
        aTground_skp,I1Tvol_skp=tom.denormalisation_teb(W,Na,Cg,Cv)
        
        print '======== {0}/{1} ========'.format(idxb,nb_ptb)
        for i,hv in enumerate(vec_hv):
            for j,sig in enumerate(vec_sig):
                J[idxb,i,j]=crit_ang(vec_gm,hv,sig,costheta,vec_kz)
                J2[idxb,i,j]=crit_ang2(vec_gm,hv,sig,costheta,vec_kz)                        
                
                Tground_skp=aTground_skp/param.get_a()
                Tvol_skp=I1Tvol_skp/param.get_I1()                
                
                Ressemb[idxb,i,j]=ressemblance(Ups,1,hv,sig,
                                               vec_gt,vec_kz,costheta,
                                               param.Na,Tvol_skp,Tground_skp)
                ErrQM[idxb,i,j]=EQM(Ups,hv,sig,
                                    vec_gt,vec_kz,costheta,
                                    param.Na,Tvol_skp,Tground_skp)
               
            
        idxminJ   = zip(*np.where(J[idxb,:,:]==np.nanmin(J[idxb,:,:])))
        idxminJ2  = zip(*np.where(J2[idxb,:,:]==np.nanmin(J2[idxb,:,:])))
        idxminMV  = zip(*np.where(Ressemb[idxb,:,:]==np.nanmin(Ressemb[idxb,:,:])))
        idxminEQM = zip(*np.where(ErrQM[idxb,:,:]==np.nanmin(ErrQM[idxb,:,:])))
        #hv                        
        hv_J[idxb]   = vec_hv[idxminJ[0][0]]
        hv_J2[idxb]  = vec_hv[idxminJ2[0][0]]        
        hv_MV[idxb]  = vec_hv[idxminMV[0][0]]
        hv_EQM[idxb] = vec_hv[idxminEQM[0][0]]
        #sigv
        sig_J[idxb]   = vec_sig[idxminJ[0][1]]
        sig_J2[idxb]  = vec_sig[idxminJ2[0][1]]
        sig_MV[idxb]  = vec_sig[idxminMV[0][1]]
        sig_EQM[idxb] = vec_sig[idxminEQM[0][1]]
        
        for m in range(Na):
            vec_gt_J[idxb,m]  = np.abs(vec_gm[m]/gammav(hv_J[idxb],sig_J[idxb],
                                                costheta,vec_kz[m]))
            vec_gt_J2[idxb,m] = np.abs(vec_gm[m]/gammav(hv_J2[idxb],sig_J2[idxb],
                                                costheta,vec_kz[m]))
            vec_gt_MV[idxb,m] = np.abs(vec_gm[m]/gammav(hv_MV[idxb],sig_MV[idxb],
                                                costheta,vec_kz[m]))
            vec_gt_EQM[idxb,m] = np.abs(vec_gm[m]/gammav(hv_EQM[idxb],sig_EQM[idxb],
                                                costheta,vec_kz[m]))
       
    return  J,J2,Ressemb,ErrQM,\
            vec_hv,vec_sig,vec_b,\
            hv_J,hv_J2,hv_MV,hv_EQM,\
            sig_J,sig_J2,sig_MV,sig_EQM,\
            vec_gt_J,vec_gt_J2,vec_gt_MV,\
            vec_gt_EQM

def estim_ecart_ang_opt(W,param,**kwargs):

    """Estimation des paramètres RVog par miminsation du critère 
    angulaire à zg connu 
    
    Attention : spécifier la valeur de zg dans kwargs.\n
    ex : estim_ecart_ang_opt(W,param,zg_connu=42)
    
    **Entrées** 
        * *W* : matrice de covariance (base )
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : contient la valeur de zg
            
    **Sorties** 
        * * *hvJ,hvJ2,hvV* : hv pour chaque estimateur
          * *sigJ,sigJ2,sigV* : sigma_v pour chaque estimateur
          * *gt_J,gt_J2,gt_V* : déco temporelle pour chaque critère
          * *X_minJ,X_minJ2,X_minV* : tableaux contenant sur chaque colonnes le resultat 
                                      de l'estimation des paramètres (hv,sigv) pour un b donné.
                                      Ceci pour chaque critère
          * *minJ,minJ2,minV* : contient la valeur du min. Ex minJ(b) = J(X_minJ)
          * *bminJ,bminJ2,bminV* : *b* minimisant la fonction de b minJ(b) pour chaque critère
          * *vec_b* : ensemble des valeur de b testées
    """
    
    start = time.time()
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    nb_ptb= 10
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    Ups = tom.MPMB_to_UPS(W)
    Ups_norm = tom.MPMB_to_UPS(W_norm)
   
    if kwargs.has_key('zg_connu'):
        zg_connu = kwargs['zg_connu']
        W_norm = tom.retranch_phig_W(zg_connu,W_norm,vec_kz)
    else:
        print 'Erreur Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
        
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)        
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    #test sur b    
    vec_b=np.linspace(interv_b[0][0],interv_b[0][1],nb_ptb)
    b_vrai=tom.b_true(R_t,param)

    #bornes de recherche du min
    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
    print 'borne hv: [{0}  {1}]'.format(hvmin,hvmax)
    print 'sig_test: [{0}  {1}]'.format(sigmin,sigmax)
    
    #X=(hv,sig) inconnu du pb d'optimisation
    X_minJ= np.zeros((2,nb_ptb),dtype='double')
    X_minJ2= np.zeros((2,nb_ptb),dtype='double')
    X_minV= np.zeros((2,nb_ptb),dtype='double')
    #valeur d'init
    X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
    #init critere J
    minJ = np.zeros(nb_ptb)
    minJ2 = np.zeros(nb_ptb)
    minV=np.zeros(nb_ptb)
    #estim des déco temp
    vec_gt_J=np.zeros((nb_ptb,Nbase))
    vec_gt_J2=np.zeros((nb_ptb,Nbase))
    vec_gt_V=np.zeros((nb_ptb,Nbase))    

    #calcul des critères angulaires et de la vraissemblance
    for idxb,b in enumerate(vec_b): 
        print '======== {0}/{1} ========'.format(idxb,nb_ptb)
        _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,b)
        vec_gm=tom.gamma_from_Rv(Rv)
        aTground_skp,I1Tvol_skp=tom.denormalisation_teb(W,Na,Cg,Cv)
        #minimisation du critère J
        xoptJ=opti.minimize(fun=J,x0=X_0,
                           args=(vec_gm,vec_kz,costheta),method ='L-BFGS-B',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)])
                      
        X_minJ[:,idxb]=xoptJ.get('x')
        minJ[idxb]=xoptJ.get('fun')
        #minimisation du critère J2        
        xoptJ2=opti.minimize(fun=J2,x0=X_0,
                           args=(vec_gm,vec_kz,costheta),method ='TNC',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)])
                                 
        X_minJ2[:,idxb]=xoptJ2.get('x')
        minJ2[idxb]=xoptJ2.get('fun')
        
        for m in range(Na):
            #fct gammav(hv,sig,costheta,vec_kz)
            vec_gt_J[idxb,m]  = np.abs(vec_gm[m]/gammav(X_minJ[0,idxb],X_minJ[1,idxb],
                                                costheta,vec_kz[m]))
            vec_gt_J2[idxb,m] = np.abs(vec_gm[m]/gammav(X_minJ2[0,idxb],X_minJ2[1,idxb],
                                                costheta,vec_kz[m]))
            vec_gt_V[idxb,m] = np.abs(vec_gm[m]/gammav(X_minV[0,idxb],X_minV[1,idxb],
                                                costheta,vec_kz[m]))
                                                
        """minimisation de la -logvraissemblance            
        temp deco : estimées avec J 
        minimisation de la -logvraissemblance"""     
        Ups_est = Ups
        
        N=param.N
        xoptV=opti.minimize(fun=mlogV,x0=X_0,
                            args=(Ups_est,vec_kz,costheta,
                                  Na,I1Tvol_skp,aTground_skp,N,vec_gt_J[idxb,:]),
                            bounds=[(hvmin,hvmax),(sigmin,sigmax)])
        
        X_minV[:,idxb]=xoptV.get('x')
        minV[idxb]=xoptV.get('fun')
        
        
    #Recuperer les minima des fonctions de b 
    idx_bmin_minJ = np.argmin(minJ)    
    idx_bmin_minJ2 = np.argmin(minJ2)    
    idx_bmin_minV = np.argmin(minV)    
    
    bminJ = vec_b[idx_bmin_minJ]
    bminJ2= vec_b[idx_bmin_minJ2]
    bminV=  vec_b[idx_bmin_minV]
    
    hvJ=X_minJ[0,idx_bmin_minJ]  
    sigJ=X_minJ[1,idx_bmin_minJ]
    gt_J=vec_gt_J[idx_bmin_minJ,:]
    
    hvJ2=X_minJ2[0,idx_bmin_minJ2]  
    sigJ2=X_minJ2[1,idx_bmin_minJ2]
    gt_J2=vec_gt_J2[idx_bmin_minJ2,:]
    
    hvV=X_minV[0,idx_bmin_minV]  
    sigV=X_minV[1,idx_bmin_minV]
    gt_V=vec_gt_V[idx_bmin_minV,:]                                                                   
    
    print 'hvJ {0} sigJ {1} hvJ2 {2} sigJ2 {3}'\
           'hvV {4} sigV {5}'.format(hvJ,sigJ,hvJ2,sigJ2,hvV,sigV)
           
    return hvJ,hvJ2,hvV,\
           sigJ,sigJ2,sigV,\
           gt_J,gt_J2,gt_V,\
           X_minJ,X_minJ2,X_minV,\
           minJ,minJ2,minV,\
           bminJ,bminJ2,bminV,\
           vec_b
           
def estim_ecart_ang_opt2(W,param,**kwargs):    
    """Estimation des paramètres RVog par miminsation des cirtères J,J2

   **Entrées** 
        * *W* : matrice de covariance (base )
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : contient la valeur de zg

    **Sorties** 
        * * *hvJ,hvJ2,hvV* : hv pour chaque estimateur
          * *sigJ,sigJ2,sigV* : sigma_v pour chaque estimateur
          * *gt_J,gt_J2,gt_V* : déco temporelle pour chaque critère
          * *X_minJ,X_minJ2,X_minV* : tableaux contenant sur chaque colonnes le resultat 
                                      de l'estimation des paramètres (hv,sigv) pour un b donné.
                                      Ceci pour chaque critère
          * *minJ,minJ2,minV* : contient la valeur du min. Ex minJ(b) = J(X_minJ)
          * *bminJ,bminJ2,bminV* : *b* minimisant la fonction de b minJ(b) pour chaque critère
          * *vec_b* : ensemble des valeur de b testées
        
    Dans cette version on utilise la recherche de l'espace (a,b)€Omega
    de positivité des matrices Rk et Ck avec la fonction search_space_definition_rob
    (robuste)
    
    """
    
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    nb_ptb= 100
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    Ups = tom.MPMB_to_UPS(W)
    Ups_norm = tom.MPMB_to_UPS(W_norm)
    #Ups_est = tom.deblanch(Ups_norm,E)
    
    #Suppression de la phase du sol    
    if kwargs.has_key('zg_connu'):
        zg_connu = kwargs['zg_connu']
        W_norm = tom.retranch_phig_W(zg_connu,W_norm,vec_kz)
    else:
        print 'estim_ecart_ang_opt2 : Erreur! Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)        
    interv_a,interv_b,_,_,approx = tom.search_space_definition_rob(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
      #interv_a,interv_b=tom.ground_selection_MB(R_t,interv_a,interv_b)        
        
    #debug : plot de la mat de covar
    """     
    pdb.set_trace()
    """     

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    #test sur b    
    vec_b=np.linspace(interv_b[0][0],interv_b[0][1],nb_ptb)
    b_vrai=tom.b_true(param)
    #vec_b=np.linspace(b_vrai-b_vrai*0.1,b_vrai+b_vrai*0.1,nb_ptb);print 'Att Debug de b: val en dur'
    #vec_b=np.array([b_vrai-b_vrai*0.1])
    #bornes de recherche du min
    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
    """
    print 'borne hv: [{0}  {1}]'.format(np.real(hvmin),np.real(hvmax))
    print 'sig_test: [{0}  {1}]'.format(sigmin,sigmax)
    """
    #X=(hv,sig) inconnu du pb d'optimisation
    X_minJ= np.zeros((2,nb_ptb),dtype='double')
    X_minJ2= np.zeros((2,nb_ptb),dtype='double')
    X_minV= np.zeros((2,nb_ptb),dtype='double')
    #valeur d'init
    X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
    #init critere J
    minJ = np.zeros(nb_ptb)
    minJ2 = np.zeros(nb_ptb)
    minV=np.zeros(nb_ptb)
    #estim des déco temp
    vec_gt_J=np.zeros((nb_ptb,Nbase))
    vec_gt_J2=np.zeros((nb_ptb,Nbase))
    vec_gt_V=np.zeros((nb_ptb,Nbase))    

    #calcul des critères angulaires et de la vraissemblance
    for idxb,b in enumerate(vec_b): 
        if not(idxb%(np.floor(nb_ptb/2))): print '======== {0}/{1} ========'.format(idxb,nb_ptb)
        _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,b)
        vec_gm=tom.gamma_from_Rv(Rv)
        aTground_skp,I1Tvol_skp=tom.denormalisation_teb(W,Na,Cg,Cv)
        #minimisation du critère J
        xoptJ=opti.minimize(fun=J,x0=X_0,
                           args=(vec_gm,vec_kz,costheta),method ='TNC',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)])
                      
        X_minJ[:,idxb]=xoptJ.get('x')
        minJ[idxb]=xoptJ.get('fun')
        #minimisation du critère J2        
        xoptJ2=opti.minimize(fun=J2,x0=X_0,
                           args=(vec_gm,vec_kz,costheta),method ='TNC',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)])
                                 
        X_minJ2[:,idxb]=xoptJ2.get('x')
        minJ2[idxb]=xoptJ2.get('fun')
        #minimisation de la -logvraissemblance            
        #temp deco : estimées avec J 
        #minimisation de la -logvraissemblance          0
        """   
        Ups_est=Ups
        N=1
        xoptV=opti.minimize(fun=mlogV,x0=X_0,
                            args=(Ups_est,vec_kz,costheta,
                                  Na,I1Tvol_skp,aTground_skp,N,vec_gt_J[idxb,:]),
                            bounds=[(hvmin,hvmax),(sigmin,sigmax)])
        
        X_minV[:,idxb]=xoptV.get('x')
        minV[idxb]=xoptV.get('fun')
        """
        for m in range(Na):
            #fct gammav(hv,sig,costheta,vec_kz)
            vec_gt_J[idxb,m]  = np.abs(vec_gm[m]/gammav(X_minJ[0,idxb],X_minJ[1,idxb],
                                                costheta,vec_kz[m]))
            vec_gt_J2[idxb,m] = np.abs(vec_gm[m]/gammav(X_minJ2[0,idxb],X_minJ2[1,idxb],
                                                costheta,vec_kz[m]))
            vec_gt_V[idxb,m] = np.abs(vec_gm[m]/gammav(X_minV[0,idxb],X_minV[1,idxb],
                                                costheta,vec_kz[m]))

    #Recuperer des minimum des fonctions de b 
    idx_bmin_minJ = np.argmin(minJ)    
    idx_bmin_minJ2 = np.argmin(minJ2)    
    idx_bmin_minV = np.argmin(minV)    
    
    hvJ=X_minJ[0,idx_bmin_minJ]  
    sigJ=X_minJ[1,idx_bmin_minJ]
    gt_J=vec_gt_J[idx_bmin_minJ,:]
    
    hvJ2=X_minJ2[0,idx_bmin_minJ2]  
    sigJ2=X_minJ2[1,idx_bmin_minJ2]
    gt_J2=vec_gt_J2[idx_bmin_minJ2,:]
    
    hvV=X_minV[0,idx_bmin_minV]  
    sigV=X_minV[1,idx_bmin_minV]
    gt_V=vec_gt_V[idx_bmin_minV,:]                                                                   

    print 'hvJ {0} sigJ {1} hvJ2 {2} sigJ2 {3}'\
           'hvV {4} sigV {5}'.format(hvJ,sigJ,hvJ2,sigJ2,hvV,sigV)
           
    return hvJ,hvJ2,hvV,\
           sigJ,sigJ2,sigV,\
           gt_J,gt_J2,gt_V,\
           X_minJ,X_minJ2,X_minV,\
           minJ,minJ2,minV,\
           vec_b
           
def estim_ecart_ang_reestim_b(W,param,**kwargs):
    """Même méthode d'estimation que estim_ecart_ang_opt
    sauf que pour chaque b, on recalcule la valeur de b
    à partir des valeur de hv et sig trouvé
    
    **Entrées** 
        * *W* : matrice de covariance (base )
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : contient la valeur de zg

    **Sorties** 
        * * *hvJ,hvJ2,hvV* : hv pour chaque estimateur
          * *sigJ,sigJ2,sigV* : sigma_v pour chaque estimateur
          * *gt_J,gt_J2,gt_V* : déco temporelle pour chaque critère
          * *X_minJ,X_minJ2,X_minV* : tableaux contenant sur chaque colonnes le resultat 
                                      de l'estimation des paramètres (hv,sigv) pour un b donné.
                                      Ceci pour chaque critère
          * *minJ,minJ2,minV* : contient la valeur du min. Ex minJ(b) = J(X_minJ)
          * *bminJ,bminJ2,bminV* : *b* minimisant la fonction de b minJ(b) pour chaque critère
          * *vec_b* : ensemble des valeur de b testées"""
    
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    nb_ptb=200
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W)
    Ups = tom.MPMB_to_UPS(W)
    Ups_norm = tom.MPMB_to_UPS(W_norm)
    #Ups_est = tom.deblanch(Ups_norm,E)
    
    #Suppression de la phase du sol    
    if kwargs.has_key('zg_connu'):
        zg_connu = kwargs['zg_connu']
        W_norm = tom.retranch_phig_W(zg_connu,W_norm,vec_kz)
    else:
        print 'Erreur Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)        
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
            
    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    #test sur b    
    vec_b=np.linspace(interv_b[0][0],interv_b[0][1],nb_ptb)
    #vec_b recalculé
    vec_br=np.zeros(nb_ptb)
    b_vrai=tom.b_true(R_t,param)
    
    #bornes de recherche du min
    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
    print 'borne hv: [{0}  {1}]'.format(hvmin,hvmax)
    print 'sig_test: [{0}  {1}]'.format(sigmin,sigmax)
    
    #X=(hv,sig) inconnu du pb d'optimisation
    X_minJ= np.zeros((2,nb_ptb),dtype='double')
    X_minJ2= np.zeros((2,nb_ptb),dtype='double')
    X_minV= np.zeros((2,nb_ptb),dtype='double')
    #valeur d'init
    X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
    #init critere J
    minJ = np.zeros(nb_ptb)
    minJ2 = np.zeros(nb_ptb)
    minV=np.zeros(nb_ptb)
    #déco temp
    vec_gt_J=np.zeros((nb_ptb,Nbase))
    vec_gt_J2=np.zeros((nb_ptb,Nbase))
    vec_gt_V=np.zeros((nb_ptb,Nbase))    
    
    #Matrice des cohérence du Volume reconstruite à partir de (hv,sig),(gti)
    vec_gvr=np.zeros((nb_ptb,Nbase),dtype='complex') #coherence reconstruites 
    vec_gvgtr=np.zeros((nb_ptb,Nbase),dtype='complex')
    Rhat_v = np.zeros((Nbase,Nbase),dtype='complex')
    
    #calcul des critères angulaires et de la vraissemblance
    for idxb,b in enumerate(vec_b): 
        print '======== {0}/{1} ========'.format(idxb,nb_ptb)
        _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,b)
        vec_gm=tom.gamma_from_Rv(Rv)
        aTground_skp,I1Tvol_skp=tom.denormalisation_teb(W,Na,Cg,Cv)
        #minimisation du critère J
        xoptJ=opti.minimize(fun=J,x0=X_0,
                           args=(vec_gm,vec_kz,costheta),method ='L-BFGS-B',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)])
                      
        X_minJ[:,idxb]=xoptJ.get('x')
        minJ[idxb]=xoptJ.get('fun')
        #minimisation du critère J2        
        xoptJ2=opti.minimize(fun=J2,x0=X_0,
                           args=(vec_gm,vec_kz,costheta),method ='TNC',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)])
                                 
        X_minJ2[:,idxb]=xoptJ2.get('x')
        minJ2[idxb]=xoptJ2.get('fun')
        
        for m in range(Nbase):
            #fct gammav(hv,sig,costheta,vec_kz)
            vec_gt_J[idxb,m]  = np.abs(vec_gm[m]/gammav(X_minJ[0,idxb],X_minJ[1,idxb],
                                                costheta,vec_kz[m]))
            vec_gt_J2[idxb,m] = np.abs(vec_gm[m]/gammav(X_minJ2[0,idxb],X_minJ2[1,idxb],
                                                costheta,vec_kz[m]))
            vec_gt_V[idxb,m] = np.abs(vec_gm[m]/gammav(X_minV[0,idxb],X_minV[1,idxb],
                                                costheta,vec_kz[m]))
        
        #Reconstruction des gv à partir des hv et sigmav estimés
        for i in range(Nbase):
            vec_gvr[idxb,i]=gammav(X_minJ[0,idxb],X_minJ[1,idxb],costheta,vec_kz[i])
            
        vec_gvgtr[idxb,:]=vec_gvr[idxb,:]*vec_gt_J[idxb,:]
        Rhat_v=tom.Rv_from_gamma(vec_gvgtr[idxb,:])
        vec_br[idxb]=tom.b_from_Rv(R_t,Rhat_v)
        

        Ups_est=Ups
        N=param.N
        xoptV=opti.minimize(fun=mlogV,x0=X_0,
                            args=(Ups_est,vec_kz,costheta,
                                  Na,I1Tvol_skp,aTground_skp,N,vec_gt_J[idxb,:]),
                            bounds=[(hvmin,hvmax),(sigmin,sigmax)])
        
        X_minV[:,idxb]=xoptV.get('x')
        minV[idxb]=xoptV.get('fun')

    #Recuperer des minimum des fonctions de b 
    idx_bmin_minJ = np.argmin(minJ)    
    idx_bmin_minJ2 = np.argmin(minJ2)    
    idx_bmin_minV = np.argmin(minV)    
    
    bminJ = vec_b[idx_bmin_minJ]
    bminJ2= vec_b[idx_bmin_minJ2]
    bminV=  vec_b[idx_bmin_minV]
 
    hvJ=X_minJ[0,idx_bmin_minJ]  
    sigJ=X_minJ[1,idx_bmin_minJ]
    gt_J=vec_gt_J[idx_bmin_minJ,:]
    
    #On recalcule hv,sig et les gti avec le b corrigé
    br=vec_br[idx_bmin_minJ]
    
    _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,br)
    vec_gm=tom.gamma_from_Rv(Rv)
    aTground_skp,I1Tvol_skp=tom.denormalisation_teb(W,Na,Cg,Cv)
    #minimisation du critère J
    xoptJr=opti.minimize(fun=J,x0=X_0,
                       args=(vec_gm,vec_kz,costheta),method ='L-BFGS-B',
                             bounds=[(hvmin,hvmax),(sigmin,sigmax)])
    X_minJr=xoptJr.get('x')       
    hvJr=X_minJr[0]
    sigJr=X_minJr[1]
    
    
    hvJ2=X_minJ2[0,idx_bmin_minJ2]  
    sigJ2=X_minJ2[1,idx_bmin_minJ2]
    gt_J2=vec_gt_J2[idx_bmin_minJ2,:]
    
    hvV=X_minV[0,idx_bmin_minV]  
    sigV=X_minV[1,idx_bmin_minV]
    gt_V=vec_gt_V[idx_bmin_minV,:]                                                                   
    
    print 'hvJ {0} sigJ {1} hvJ2 {2} sigJ2 {3}'\
           'hvV {4} sigV {5}'.format(hvJ,sigJ,hvJ2,sigJ2,hvV,sigV)
    print 'hvJr {0} sigJr{1}'.format(hvJr,sigJr)
    return hvJ,hvJ2,hvV,\
           sigJ,sigJ2,sigV,\
           gt_J,gt_J2,gt_V,\
           X_minJ,X_minJ2,X_minV,\
           minJ,minJ2,minV,\
           bminJ,bminJ2,bminV,\
           vec_b,vec_br
           
def estim_ecart_ang_tot(W,param,**kwargs):    
    """Estimation des paramètres RVog par miminsation du critère J_tot
    on minimise J(b,hv,sigv) par rapport à b,hv,sigv 
    Dans cette version on utilise la recherche de l'espace (a,b)€Omega
    de positivité des matrices Rk et Ck avec la fonction robuste
    
    **Entrées** 
        * *W* : matrice de covariance (base )
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : contient la valeur de zg    
    
    **Sortie**
        * *btot,hvJtot,sigJtot* : Resultats d'estimation
    """
    
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    nb_ptb= 100
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    Ups = tom.MPMB_to_UPS(W)
    Ups_norm = tom.MPMB_to_UPS(W_norm)
    
    #Suppression de la phase du sol    
    if kwargs.has_key('zg_connu'):
        zg_connu = kwargs['zg_connu']
        W_norm = tom.retranch_phig_W(zg_connu,W_norm,vec_kz)
    else:
        print 'Erreur Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
    
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)            
    interv_a,interv_b,_,_ = tom.search_space_definition(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    b_vrai=tom.b_true(R_t,param)

    #bornes de recherche du min
    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
    bmin=np.min(interv_b[0])
    bmax=np.max(interv_b[0])

    #valeur d'init
    X_0 = np.array([(bmin+bmax)/2,(hvmin+hvmax)/2,(sigmin+sigmax)/2])
    
    #minimisation du critère J_total
    xoptJ_tot=opti.minimize(fun=J_tot,x0=X_0,
                       args=(R_t,vec_kz,costheta),method ='TNC',
                             bounds=[(bmin,bmax),(hvmin,hvmax),(sigmin,sigmax)])
                  
    X_minJ_tot=xoptJ_tot.get('x')
    btot=X_minJ_tot[0]
    hvJtot=X_minJ_tot[1]
    sigJtot=X_minJ_tot[2]        
    
    print 'btot {0} hvJtot {1} sigJtot {2} '.format(btot,hvJtot,sigJtot)
    return btot,hvJtot,sigJtot
        
def estim_ecart_ang_scal(W,param,**kwargs):
    """Estimation des paramètres RVog par miminsation du critère J2_scal.
    Minise J2 par rapport a (hv,sig) puis par rapport à b.
    
    **Entrées** 
        * *W* : matrice de covariance (base )
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : contient la valeur de zg    
            * *kwargs['U0']* : initial guess pour (hv,sigv)
            * *kwargs['b0']* : initial guess pour b
            
    **Sortie**
        * *hv,sigv,vec_gt,bopt* : resultats d'estimation
    """

    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))    
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    Ups = tom.MPMB_to_UPS(W)
    Ups_norm = tom.MPMB_to_UPS(W_norm)
    
    #Suppression de la phase du sol    
    if kwargs.has_key('zg_connu'):
        zg_connu = kwargs['zg_connu']
        W_norm = tom.retranch_phig_W(zg_connu,W_norm,vec_kz)
    else:
        print 'Erreur Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
        
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)                    
    interv_a,interv_b,_,_,_ = tom.search_space_definition_rob(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    

    #pdb.set_trace()
    if kwargs.has_key('U0'):
        U0 = kwargs['U0']
    else:  
        sigmin=0.001
        sigmax=0.1
        hvmin=5.
        hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité        
        U0= np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
        
    #bornes de recherche du min sur b      
    bmin=np.min(interv_b[0][0])
    bmax=np.max(interv_b[0][1])

    if kwargs.has_key('b0'):
        b0=kwargs['b0']
    else:
        b0=0.5*(bmin+bmax)    

    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
    
    if kwargs.has_key('critere'):
        if kwargs['critere']=='J2':

            xoptscal=opti.fmin(func=J2_scal,
                               args=(R_t,vec_kz,costheta,U0),x0=b0)
            bopt=xoptscal
            Rv=bopt*R_t[0]+(1-bopt)*R_t[1]
            vec_gm=tom.gamma_from_Rv(Rv)
            #X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
            xoptJ=opti.minimize(fun=J2,x0=U0,
                                args=(vec_gm,vec_kz,costheta),method ='TNC',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)],
                                         options={'xtol':10**-8})
            
        elif kwargs['critere']=='J':

            xoptscal=opti.fmin(func=J_scal,
                               args=(R_t,vec_kz,costheta,U0),x0=b0)
            bopt=xoptscal
            Rv=bopt*R_t[0]+(1-bopt)*R_t[1]
            vec_gm=tom.gamma_from_Rv(Rv)
            #X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
            xoptJ=opti.minimize(fun=J,x0=U0,
                                args=(vec_gm,vec_kz,costheta),method ='TNC',
                                 bounds=[(hvmin,hvmax),(sigmin,sigmax)],
                                         options={'xtol':10**-8})
                              
            
        else:
            print 'Erreur critère non reconnu'
            sys.exit(1)
    else:        
        xoptscal=opti.minimize_scalar(fun=J2_scal,
                                      args=(R_t,vec_kz,costheta,U0),
                                      method ='Bounded',bounds=(bmin,bmax),
                                      options={'xatol':10**-8})
        bopt=xoptscal
        Rv=bopt*R_t[0]+(1-bopt)*R_t[1]
        vec_gm=tom.gamma_from_Rv(Rv)
        #X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
        xoptJ=opti.minimize(fun=J2,x0=U0,
                            args=(vec_gm,vec_kz,costheta),method ='TNC',
                            bounds=[(hvmin,hvmax),(sigmin,sigmax)],
                            options={'xtol':10**-8})
    
    X_minJ=xoptJ.get('x')
    hv=X_minJ[0]
    sigv=X_minJ[1]
    vec_gt = np.zeros(Nbase)
    for i in range(Nbase):
        vec_gt[i] = vec_gm[i]/gammav(hv,sigv,costheta,vec_kz[i])

    return hv,sigv,vec_gt,bopt
    
def estim_V_scal(W,param,**kwargs):
    """Estimation des paramètres RVog par miminsation de la vraissemblance
    sur chaque b
    
    **Entrées** 
        * *W* : matrice de covariance (base )
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : contient la valeur de zg    
            
    **Sortie**
        * *hv,sigv,bopt* : resultats d'estimation
    """
    
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    N=param.N#Taille d'echant
    
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    vec_gt = param.get_gtlist()#ex pour Na=3 gt12 gt13 gt23 

    #Ups_est = tom.deblanch(Ups_norm,E)
    #Suppression de la phase du sol    
    if kwargs.has_key('zg_connu'):
        zg = kwargs['zg_connu']
        W = tom.retranch_phig_W(zg,W,vec_kz)
        param.z_g=0
    else:
        print 'Erreur Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
        
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)        
    interv_a,interv_b,_,_,_ = tom.search_space_definition_rob(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]
    
    #bornes de recherche du min    
    b_vrai=tom.b_true2(R_t,param)
    bmin=np.min(interv_b[0])
    bmax=np.max(interv_b[0])
    #pdb.set_trace()
    xoptscal=opti.minimize_scalar(fun=V_scal,
                        args=(W_norm,R_t,C_t,vec_kz,costheta,a,Na,N,vec_gt,param.z_g),method ='Bounded',
                        bounds=(bmin,bmax),options={'xatol':10**-8})
    bopt=xoptscal.get('x')  
    
    #extraction des hv et sig correspondant au min de J(b)
    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité

    #Rv=bopt*R_t[0]+(1-bopt)*R_t[1]
    #vec_gm=tom.gamma_from_Rv(Rv)
    X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
    
    
    _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,bopt)
    aTground,I1Tvol=tom.denormalisation_teb(W,Na,Cg,Cv)        
    Ups_est=tom.MPMB_to_UPS(W)
    X_0 = np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2])
    xoptV = opti.minimize(fun=mlogV,x0=X_0,\
                        args=(Ups_est,vec_kz,costheta,Na,I1Tvol,aTground,N,vec_gt,param.z_g),\
                        method ='TNC',bounds=[(hvmin,hvmax),(sigmin,sigmax)],\
                        options={'ftol':10**-8})
        
    X_minV=xoptV.get('x')
    hv=X_minV[0]
    sigv=X_minV[1]
    return hv,sigv,bopt
    
def estim_mlogV_zg_known(W,param,**kwargs):
    """Estimation des paramètres RVog à par miminsation de la -logvraissemblance à zg connu.
    
    eta = (Tvol,Tground,hv,sigv,{gt}ij). {gt}ij ensemble des 
    decohérence temporelle tq i<j. Ex en Dual baseline gt12,gt13,gt23
    
    **Entrées** 
        * *W* : matrice de covariance (base )
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['U0']* : initial guess pour (hv,sigv)
            
    **Sortie**
        * *Tvol,Tground,hv,sigv,vec_gt* : resultats d'estimation
        * *reu* : scalaire indiquant si la minimisation du critère a convergée 
    """  
    
    Ngt = mb.get_Nb_from_Na(param.Na)
    
    if kwargs.has_key('U0'):        
        X0 = kwargs['U0']
    else:
        print 'estim_mlogV_zg_knwon X0 inconnu !'


    Ups_n = tom.MPMB_to_UPS(W)

    verbose=0
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
          
    X = opti.minimize(fun = mlogV_zg_known,
                      method='SLSQP',
                      x0=X0,
                      options={'xtol':1e-10,'ftol':1e-10,
                               'disp':1,'iprint':1,
                                'maxiter':500},
                      bounds=bds,
                      args=arg) 
           
    Xopt =  X['x'][0] #Mdofi Capdessus : X['x'] contient [x,list_x,list_fx]
    vec_Tvol = Xopt[0:9]
    if X['status'] in [0,9] and X['fun'] != 1e14:
        #L'optimisation a convergé si le status est 0 (f(xk+1)-f(xk)<ftol)
        #ET si il n'y pas convergence sur le plateau a 1e14
        reu=X['status']
    else:
        reu=np.inf
        print 'pas converge : {0} f(X)={1}'.format(X['status'],X['fun'])
        
    Tvol = mb.get_Tm_from_vecTm(vec_Tvol)
    vec_Tground = Xopt[9:18]
    Tground = mb.get_Tm_from_vecTm(vec_Tground)
    hv = Xopt[18]
    sigv = Xopt[19]
    
    if 20+Ngt != len(Xopt):
        print 'estim_mlogV_zg_known: pb taille Xopt'
    else:        
        vec_gt = Xopt[20:20+Ngt]        
    return Tvol,Tground,hv,sigv,vec_gt,reu
    
def estim_mlogV_gt_inconnu_all(W,param,**kwargs):
    """Estimation des paramètres RVog à par miminsation 
    de la vraissemblance sur chaque fonction de b,hv,sigv,{gt} 
    d'un bloc (all)
    
    **Entrées** 
        * *W* : matrice de covariance (base)
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : valeur de zg
            * *kwargs['U0']* : initial guess pour (hv,sigv)
            
    **Sortie**
        * *hv,sigv,vec_gt,bopt* : resultats d'estimation        
    """
    
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    N=param.N#Taille d'echant
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 

    #Suppression de la phase du sol    
    if kwargs.has_key('zg_connu'):
        zg_connu = kwargs['zg_connu']
        W = tom.retranch_phig_W(zg_connu,W,vec_kz)
        param.z_g=0
    else:
        print 'Erreur Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)    
    interv_a,interv_b,_,_,_ = tom.search_space_definition_rob(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    #bornes de recherche du min
    b_vrai = tom.b_true2(R_t,param)        
    
    bmin=np.min(interv_b[0])
    bmax=np.max(interv_b[0])

    sigmin=0.01
    sigmax=0.1
    
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité

    Ngt=mb.get_Nb_from_Na(Na)    
    X_0 = np.hstack((np.array([(bmin+bmax)/2,
                               (hvmin+hvmax)/2,
                                (sigmin+sigmax)/2]),np.ones(Ngt)))
    
    xoptscal=opti.minimize(fun=mlogV_gt_inconnu_all,x0=X_0,
                           args=(W_norm,R_t,C_t,vec_kz,costheta,a,Na,N,param.z_g),
                           method ='TNC',
                           bounds=[(bmin,bmax),(hvmin,hvmax),(sigmin,sigmax)]+[(0,1)]*Ngt,                     
                           options={'xtol':10**-100})
      
    
    X = xoptscal.get('x')  
    bopt = X[0]
    hv = X[1]
    sigv = X[2]
    vec_gt = X[3:]        
    return hv,sigv,vec_gt,bopt
    
def estim_V_scal_gt_inconnu(W,param,**kwargs):
    """Estimation des paramètres RVog à gt connu par miminsation 
    de la vraissemblance sur chaque b
    
    **Entrées** 
        * *W* : matrice de covariance (base)
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['zg_connu']* : valeur de zg
            
    **Sortie**
        * *hv,sigv,vec_gt,bopt* : resultats d'estimation"""
        
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    Nbase = int(np.floor(Na*(Na-1)/2))
    N=param.N#Taille d'echant
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 

    #Suppression de la phase du sol    
    if kwargs.has_key('zg_connu'):
        zg_connu = kwargs['zg_connu']
        W = tom.retranch_phig_W(zg_connu,W,vec_kz)
        param.z_g=0
    else:
        print 'Erreur Specifier un zg (technique a zg inconnu indisponible)'
        sys.exit(1)
    
    W_norm,E = tom.normalize_MPMB_PS_Tebald(W,Na)
    #SKP
    R_t,C_t,_=tom.sm_separation(W_norm,Np,Na)    
    interv_a,interv_b,_,_,_ = tom.search_space_definition_rob(R_t,C_t,Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    #bornes de recherche du min
    b_vrai=tom.b_true2(R_t,param)        
    bmin=np.min(interv_b[0])
    bmax=np.max(interv_b[0])
    #pdb.set_trace()
    xoptscal=opti.minimize_scalar(fun=V_scal_gt_inconnu,
                                  args=(W_norm,R_t,C_t,vec_kz,costheta,a,Na,N,param.z_g),
                                  method ='Bounded',bounds=(bmin,bmax),
                                  options={'xatol':-10**-100})
    
    bopt=xoptscal.get('x')  
    print 'fin determ de b. bopt : {0}'.format(bopt)
    #raw_input()
    #pdb.set_trace()    
    #extraction des hv et sig correspondant au min de v_scal_gt_inconnu
    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
        
    Ngt=mb.get_Nb_from_Na(Na)
    _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,bopt)
    aTground,I1Tvol=tom.denormalisation_teb(W,Na,Cg,Cv)    
    Ups_est=tom.MPMB_to_UPS(W)
    #ex en DualB: X_0=(hv_0,sig_0,gt12_0,gt13_0,gt23_0)
    X_0 = np.hstack((np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2]),np.ones(Ngt)))
        
    xoptV=opti.minimize(fun=mlogV_gt_inconnu,x0=X_0,
                        args=(Ups_est,vec_kz,costheta,Na,I1Tvol,aTground,N,param.z_g),
                        method ='TNC',
                        bounds=[(hvmin,hvmax),(sigmin,sigmax)]+[(0,1)]*Ngt,
                        options={'xtol':10**-8})
    
    X_minV=xoptV.get('x')
    hv=X_minV[0]
    sigv=X_minV[1]
    vec_gt=X_minV[2:]
    
    return hv,sigv,vec_gt,bopt

def estim_mlogV_Tg_Tv_zg_known(W,param,**kwargs):
    """Estimation des paramètres RVoG (hv,sigv,gt) par miminsation 
    de la vraissemblance (Tvol et Tground connu)
    
    **Entrées** 
        * *W* : matrice de covariance (base)
        * *param* : object de type param_rvog
        * *\**kwargs* : dictionnaire d'option
            * *kwargs['U0']* : initial guess pour (hv,sigv,{gt}ij)
            
    **Sortie**
        * *hv,sigv,vec_gt* : resultats d'estimation"""
    
    Np=3
    Na=int(np.floor(W.shape[0]/Np))
    #param 
    costheta=np.cos(param.theta)
    vec_kz = param.get_kzlist() #ex pour Na=3 kz12 kz13 kz23 
    Ups = tom.MPMB_to_UPS(W)
    Ngt=mb.get_Nb_from_Na(Na)  

    sigmin=0.01
    sigmax=0.1
    hvmin=5.
    hvmax=2.*np.pi/np.max(vec_kz)#le min des hauteurs d'ambiguité
   
    if kwargs.has_key('U0'):
        U0 = kwargs['U0']
    else:                
        U0 = np.hstack((np.array([(hvmin+hvmax)/2,(sigmin+sigmax)/2]),0.5*np.ones(Ngt)))
        
    
    xopt=opti.minimize(fun=mlogV_Tg_Tv_zg_known,x0=U0,
                        args=(Ups,vec_kz,costheta,Na,param.T_vol,param.T_ground,param.N,param.z_g),
                        method ='TNC',
                        bounds=[(hvmin,hvmax),(sigmin,sigmax)]+[(0,1)]*Ngt,
                        options={'xtol':10**-11,'maxiter':500})
    
    X_min=xopt.get('x')
    hv=X_min[0]
    sigv=X_min[1]
    vec_gt=X_min[2:]
    return hv,sigv,vec_gt
        
def mlogV_zg_known(X,*arg):    
    """Calcul du critère de -logV en fonction des var : Tvol,Tground,hv,sigma_v,{gt}ij
    
    **NB**: Tvol et Tground stockées sous forme de vecteur (cf :ref:`Annexe <vecto>`)
    
    **Entrées**
        * *X* : X = (Tvol,Tground,hv,sigma_v,{gt}ij)
        * *arg* : paramètres necessaires au calcul de mlogV_zg_known
            * *arg[0]* : Ups_est : matrice de covariance empirique (base lexico)
            * *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij, i<j}                        
            * *arg[2]* : costheta : cos (angle incidence)                  
            * *arg[3]* : Na : nombre d'antennes                  
            * *arg[4]* : N : taille de l'echantillon                
            * *arg[5]* : zg : altitude du sol 
            * *arg[6]* : disp : booleen, mode affichage
            
    **Sortie**
        * *V* : valeure du critère 
    """
    
    Ups_est=arg[0]
    vec_kz=arg[1]
    costheta=arg[2]
    Na=arg[3]
    N=arg[4]
    zg=arg[5]
    if len(arg)==7:
        disp = arg[6]
    else:
        disp = 0
    
    if len(arg)==0:
         print '=== ! Erreur Manque des paramètrees d\entré bro ! ==='
    else:
         nbarg=len(X)         
         vec_Tvol=X[0:9]
         vec_Tground=X[9:18]
         Tvol = mb.get_Tm_from_vecTm(vec_Tvol)
         Tground = mb.get_Tm_from_vecTm(vec_Tground)         
         hv=X[18]
         sig=X[19]         
         vec_gt = np.zeros((nbarg-20,1),dtype='float')
         for i in range(len(vec_gt)):                          
             vec_gt[i] = np.real(X[i+20])        

         eigTvol,_ = npl.eig(Tvol)
         eigTground,_ = npl.eig(Tground)
         
         seuilneg = -1e-6
         seuilimag = 1e-10
         value_inf = 1e14
         c = 0
         Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg)
         if np.sum(np.isnan(Ups_mod))>1 or np.sum(np.isinf(Ups_mod))>1 :             
             if disp : print 'mlogV({0}) : Ups_mod=nan/inf'.format(c);c +=1 
             return value_inf
         else:
             eigUps,_ = npl.eig(Ups_mod)                 
                 
             if np.min(np.real(eigUps)) < seuilneg:
                 if disp : 
                     print 'mlogV({0}) : Ups nég/sing!'.format(c)
                     print ' '.join(['{:04.3f}+{:04.3f}j'
                               .format(float(np.real(eigUps[i])),
                                       float(np.imag(eigUps[i])))
                                       for i in range(len(eigUps))])                     
                     c +=1 
                 return value_inf 
                 
             if np.min(np.imag(eigUps)) > seuilimag:
                 if disp : print 'mlogV({0}) : v.a.p de Ups non relles'.format(c); c +=1 
                 return value_inf 
                              
         if np.sum(vec_gt>1)>=1:             
             if disp : print 'mlogV({0}) : gt>1'.format(c); c +=1 
             return value_inf 
                 
         if np.min(np.real(eigTvol))<seuilneg or \
            np.min(np.real(eigTground))< seuilneg:                
                #Tvol ou Tground non semi-def-postivie !
                if disp :                    
                    print 'mlogV({0}) : Tvol ou Tgro non positv!'.format(c)
                    c +=1               
                return value_inf 
             
         if np.min(np.imag(eigTvol))>seuilimag or \
            np.min(np.imag(eigTvol))>seuilimag:
            if disp : print 'mlogV({0}) : Tvol ou Tgro non reelle!'.format(c); c +=1 
            return value_inf 
                 
         invUps = bl.inv_cond(Ups_mod,return_type='inf',cond_max=1e14)         
         if np.isinf(bl.inv_cond(Ups_mod,return_type='inf')[0,0]):
             if disp : print 'mlogV({0}) : Ups singuliere'.format(c); c +=1 
             #Matrice Ups_mod singulière
             V = value_inf
         else:
             #V = np.real(N*(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
             #    np.trace(Ups_est.dot(invUps))))
             #argmin(N*f) = argmin(f)
             V = np.real((np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
                 np.trace(Ups_est.dot(invUps))))
                 
    return V
    
def mlogV_zg_known_noconstraint(X,*arg):        
    """Calcul du critère de -logV en fonction des 2 var :  hv, sig
    
    **NB**: Tvol et Tground stockées sous forme de vecteur (cf :ref:`Annexe <vecto>`)
    
    **Entrées**
        * *X* : X = (Tvol,Tground,hv,sigma_v,{gt}ij)
            * *arg* : paramètres necessaires au calcul de J_tot
                * *arg[0]* : Ups_est : matrice de covariance empirique (base lexico)
                * *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}
                * *arg[2]* : costheta : cos (angle incidence)                  
                * *arg[3]* : Na : nombre d'antennes                  
                * *arg[4]* : N : taille de l'echantillon 
                * *arg[5]* : zg : altitude du sol 
                * *arg[6]* : disp : booleen, mode affichage
    **Sortie** 
        * *V* : critère -logV                
    """
    
    Ups_est=arg[0]
    vec_kz=arg[1]
    costheta=arg[2]
    Na=arg[3]
    N=arg[4]
    zg=arg[5]
    if len(arg)==7:
        disp = arg[6]
    else:
        disp = 0
    
    if len(arg)==0:
         print '=== ! Erreur Manque des paramètrees d\entré bro ! ==='
    else:
         nbarg=len(X)         
         vec_Tvol=X[0:9]
         vec_Tground=X[9:18]
         Tvol = mb.get_Tm_from_vecTm(vec_Tvol)
         Tground = mb.get_Tm_from_vecTm(vec_Tground)         
         hv=X[18]
         sig=X[19]         
         vec_gt = np.zeros((nbarg-20,1),dtype='float')
         for i in range(len(vec_gt)):                          
             vec_gt[i] = np.real(X[i+20])
         

         eigTvol,_ = npl.eig(Tvol)
         eigTground,_ = npl.eig(Tground)
         
         seuilneg = -1e-6
         seuilimag = 1e-10
         value_inf = 1e14
         c = 0
         Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg)
         invUps = bl.inv_cond(Ups_mod,return_type='inf',cond_max=1e14)  
         V = np.real((np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
                     np.trace(Ups_est.dot(invUps))))
    return V             
    
def mlogV(X,*arg):
    """Calcul du critère de -logV en fonction des vars : Tvol,Tground,hv,sigm_v,{gt}ij

    **NB**: Tvol et Tground stockées sous forme de vecteur (cf :ref:`Annexe <vecto>`)
    
    **Entrées**
        * *X* : X = (Tvol,Tground,hv,sigma_v,{gt}ij)
        * *arg* : paramètres necessaires au calcul de mlogV_zg_known
            * *arg[0]* : Ups_est : matrice de covariance empirique (base lexico)
            * *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}                        
            * *arg[2]* : costheta : cos (angle incidence)                  
            * *arg[3]* : Na : nombre d'antennes                  
            * *arg[4]* : I1Tvol : paramètres RVoG necessaire a la reconstruction(correspond à Cv de la décomp SKP)                          
            * *arg[5]* : aTground : paramètres RVoG  la reconstruction (correspond à Cg de la décomp SKP)
            * *arg[6]* : N : taille de l'echantillon 
            * *arg[7]* : vec_gt, ensemble des décorrélation temporelles dans l'ordre {gtij , i<j}            
            * *arg[8]* : zg : altitude du sol         
            
    **Sortie**
        * *V* : valeure du critère"""       
        
    Ups_est=arg[0]
    vec_kz=arg[1]
    costheta=arg[2]
    Na=arg[3]
    I1Tvol=arg[4]
    aTground=arg[5]
    N=arg[6]
    vec_gt=arg[7]
    zg=arg[8]
    
    if len(arg)==0:
        print '=== ! Erreur Manque des paramètrees d\entré bro ! ==='
        return np.nan
    else:
        hv=X[0]
        sig=X[1]
        Ups_mod = Ups_reconstr2(hv,sig,vec_gt,vec_kz,costheta,Na,I1Tvol,aTground,zg)
        V = np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
            N*np.trace(Ups_est.dot(npl.inv(Ups_mod))))
        
        return V
        
def mlogV_Tg_Tv_zg_known(X,*arg):
    """Calcul du critère de -logV en fonction des vars : hv, sig,{gt}ij .
    
    **Entrées**
        * *X* : X = (hv,sigma_v,{gt}) avec {gt} ensemble des decohé temp placé dans l'ordre i<j, ex en DualB : gt12,gt13,gt23
        * *arg* : paramètres necessaires au calcul de mlogV_Tg_Tv_zg_known
            * *arg[0]* : Ups_est : matrice de covariance empirique (base lexico)
            * *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}
            * *arg[2]* : costheta : cos (angle incidence)                  
            * *arg[3]* : Na : nombre d'antennes                  
            * *arg[4]* : Tvol                   
            * *arg[5]* : Tground                           
            * *arg[6]* : N : taille de l'echantillon 
            * *arg[7]* : zg altitude du sol
    **Sortie**
        * *V* : valeure du critère"""    

    Ups_est = arg[0]
    vec_kz = arg[1]
    costheta = arg[2]
    Na = arg[3]
    Tvol = arg[4]
    Tground = arg[5]
    N = arg[6]
    zg = arg[7]
    
    if len(arg)==0:
        print '=== ! Erreur Manque des paramètrees d\entré bro ! ==='
        return np.nan
    else:
        #pdb.set_trace()
        #print 'mlogV_Tg_Tv_known X={0}'.format(X)
        nbarg=len(X)
        hv=X[0]
        sig=X[1]
        vec_gt = np.zeros((nbarg-2,1),dtype='float')
        for i in range(len(vec_gt)):
            vec_gt[i]=X[i+2]
        Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg)
        
        try:       
            V = N*(np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
                np.trace(Ups_est.dot(npl.inv(Ups_mod)))))
                
        except:
            print 'mlogV_gt_inconnu: pb de calcul du critère (inf renvoye)'
            return 10**100#renvoyer inf semble faire planter
            
        return V

def mlogV_Tg_Tv_zg_sigv_known(X,*arg):
    """Calcul du critère de -logV en fonction des 2 var :  hv, {gt}ij
    
    **Entrées**
        * *X* : X = (hv,{gt}) avec {gt} ensemble des decohé temp placé dans l'ordre i<j\n
                ex en DualB : gt12,gt13,gt23
        * *arg* : paramètres necessaires au calcul de mlogV_Tg_Tv_zg_sigv_known
            * *arg[0]* : Ups_est : matrice de covariance empirique (base lexico)
            * *arg[1]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}                          
            * *arg[2]* : costheta : cos (angle incidence)                  
            * *arg[3]* : Na : nombre d'antennes                  
            * *arg[4]* : Tvol                   
            * *arg[5]* : Tground                           
            * *arg[6]* : N : taille de l'echantillon 
            * *arg[7]* : zg altitude du sol
            * *arg[8]* : sigma_v
    **Sortie**
        * *V* : valeure du critère"""    

    Ups_est = arg[0]
    vec_kz = arg[1]
    costheta = arg[2]
    Na = arg[3]
    Tvol = arg[4]
    Tground = arg[5]
    N = arg[6]
    zg = arg[7]
    sig = arg[8]
    
    if len(arg)==0:
        print '=== ! Erreur Manque des paramètrees d\entré bro ! ==='
        return np.nan
    else:
        #pdb.set_trace()
        #print 'mlogV_Tg_Tv_known X={0}'.format(X)
        nbarg=len(X)
        hv=X[0]
        vec_gt = np.zeros((nbarg-1,1),dtype='float')
        for i in range(len(vec_gt)):
            vec_gt[i]=X[i+1]
            
        Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg)
        
        try:       
            V = N*(np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
                np.trace(Ups_est.dot(npl.inv(Ups_mod)))))
                
        except:
            print 'mlogV_gt_inconnu: pb de calcul du critère (inf renvoye)'
            return 10**100#renvoyer inf semble faire planter
            
        return V

def mlogV_gt_inconnu_all(X,*arg):
    """Calcul du critère de -logV en fonction des var :  b, hv, sigv,{gt}.
    
    **Entrées**
        * *X* : X = (b,hv,sigma_v,{gt}) avec {gt} ensemble des decohé temp placé dans l'ordre i<j\n
                ex en DualB : gt12,gt13,gt23
        * *arg* : paramètres necessaires au calcul de J_tot
            * *arg[0]* : W : matrice de covariance empirique (base MPMB)
            * *arg[1]* : liste de matrices de structure R_t (SKP)
            * *arg[2]* : liste de matrices de rep polar C_t (SKP)
            * *arg[3]* : vec_kz, ensemble des baseline dans l'ordre {kzij , i<j}                           
            * *arg[4]* : costheta : cos (angle incidence)
            * *arg[5]* : a (parametre SKP)
            * *arg[6]* : Na : nombre d'antennes                                          
            * *arg[7]* : N : taille de l'echantillon 
            * *arg[8]* : zg altitude du sol                    

    **Sortie**
        * *V* : valeure du critère"""    

    W = arg[0]
    R_t = arg[1]
    C_t = arg[2]    
    vec_kz = arg[3]
    costheta = arg[4]
    a = arg[5]    
    Na = arg[6]
    N = arg[7]
    zg = arg[8]
    
    if len(arg)==0:
        print '=== ! Erreur Manque des paramètrees d\entré bro ! ==='
        return np.nan
    else:
        nbarg=len(X)
        b=X[0]
        hv=X[1]
        sig=X[2]
        vec_gt = np.zeros((nbarg-3,1),dtype='float')
        for i in range(len(vec_gt)):
            vec_gt[i]=X[i+2]
                    
        _,Rg,Rv,Cg,Cv = tom.value_R_C(R_t,C_t,a,b)
        aTground,I1Tvol=tom.denormalisation_teb(W,Na,Cg,Cv)
        Ups_est=tom.MPMB_to_UPS(W)
                
        Ups_mod = Ups_reconstr2(hv,sig,vec_gt,vec_kz,costheta,Na,I1Tvol,aTground,zg)
        
        try:   
            V = np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
                N*np.trace(Ups_est.dot(npl.inv(Ups_mod))))
            if V<0:
                print 'Critère mlogV_gt_inconnu_all négatif !!'
                #pdb.set_trace()
                return np.inf
        except:
            print 'mlogV_gt_inconnu: pb de calcul du critère (inf renvoye)'
            return np.inf
            
        #print '->>> hv : {0} sig : {1} gt12 : {2} gt13 : {3} gt23 : {4}'.format(hv,sig,vec_gt[0],vec_gt[1],vec_gt[2])
        #print '->>>>hv: {0} sig: {1} gt12: {2} gt13: {3} gt23: {4} V: {5}'.format(hv,sig,vec_gt[0],vec_gt[1],vec_gt[2],V) 
        return V

   
def mlogV_hv_sig_gt_Tg_zg_known(X,*arg):
    """Calcul du critère de -logV lorsque tous les paramètres sont connus (Tground,hv,sigv,zg,gt12,gt13,gt23) 
    sauf Tvol. Cette donc une fonction des 9 coeff de Tvol.
        
    **Entrées**
       * *X* : X = (coeff de Tvol placé dans l'ordre définit par A.Arnaubec
                    dans sa thèse, cf :ref:`Annexe <vecto>`)
       * *arg* : paramètres necessaires au calcul 
                  arg[0] : Ups : matrice de covariance empirique 
                          (base lexico)
                  arg[1] : vec_kz, ensemble des baseline dans l'ordre 
                           {kzij , i<j}
                           
                  arg[2] : costheta : cos (angle incidence)                  
                  arg[3] : Na : nombre d'antennes                  
                  arg[4] : hv : hauteur de vegetation
                  arg[5] : Tground                           
                  arg[6] : N : taille de l'echantillon 
                  arg[7] : zg altitude du sol               
                  arg[8] : sigv : attenuation dans le vol. de véget.
                  arg[9] : vec_gt ensemble des gt : gt12 gt13 gt23
    """
    if len(arg) != 10:
        print '=== ! Erreur des paramètrees d\entré ! ==='
        return np.nan
    else:
        nbarg = len(arg)
        Ups_est = arg[0]
        vec_kz = arg[1]
        costheta = arg[2]
        Na = arg[3]
        hv = arg[4]    
        Tground = arg[5]
        N = arg[6]
        zg = arg[7]
        sig = arg[8]
        vec_gt = arg[9]
        
        Tvol = mb.get_Tm_from_vecTm(X)        
            
        Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg)
                 
        V = N*(np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
               np.trace(Ups_est.dot(npl.inv(Ups_mod)))))

        return V

def mlogV_hv_sig_gt_Tv_zg_known(X,*arg):
    """Calcul du critère de -logV lorsque tous les paramètres sont connus (Tvol,hv,sigv,zg,gt12,gt13,gt23) 
    sauf Tground. Cette donc une fonction des 9 coeff de Tground.
    
    **Entrées**
       * *X* : X = (coeff de Tground placé dans l'ordre définit par A.Arnaubec
                    dans sa thèse)
       * *arg* : paramètres necessaires au calcul 
                  arg[0] : Ups : matrice de covariance empirique 
                          (base lexico)
                  arg[1] : vec_kz, ensemble des baseline dans l'ordre 
                           {kzij , i<j}
                           
                  arg[2] : costheta : cos (angle incidence)                  
                  arg[3] : Na : nombre d'antennes                  
                  arg[4] : hv : hauteur de vegetation
                  arg[5] : Tground                           
                  arg[6] : N : taille de l'echantillon 
                  arg[7] : zg altitude du sol               
                  arg[8] : sigv : attenuation dans le vol. de véget.
                  arg[9] : vec_gt ensemble des gt : gt12 gt13 gt23
    """
    if len(arg) != 10:
        print '=== ! Erreur des paramètrees d\entré ! ==='
        return np.nan
    else:
        nbarg = len(arg)
        Ups_est = arg[0]
        vec_kz = arg[1]
        costheta = arg[2]
        Na = arg[3]
        hv = arg[4]    
        Tvol = arg[5]
        N = arg[6]
        zg = arg[7]
        sig = arg[8]
        vec_gt = arg[9]
        
        Tground = mb.get_Tm_from_vecTm(X)        
            
        Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg)
                 
        V = N*(np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
               np.trace(Ups_est.dot(npl.inv(Ups_mod)))))

        return V    
        
def mlogV_hv_sig_gt_Tv_Tg_zg_known(X,*arg):
    """Calcul du critère de -logV lorsque tous les paramètres sont connus (Tvol,Tg,hv,sigv,zg) 
    sauf les decohe. temporelles.
    Cette donc une fonction de gt12 gt13 gt23 (cas DB)
    
    **Entrées**
       * *X* : X = (g12,gt13,gt23)
       * *arg* : paramètres necessaires au calcul 
                  arg[0] : Ups : matrice de covariance empirique 
                          (base lexico)
                  arg[1] : vec_kz, ensemble des baseline dans l'ordre 
                           {kzij , i<j}
                           
                  arg[2] : costheta : cos (angle incidence)                  
                  arg[3] : Na : nombre d'antennes                  
                  arg[4] : hv : hauteur de vegetation
                  arg[5] : Tground                           
                  arg[6] : N : taille de l'echantillon 
                  arg[7] : zg altitude du sol               
                  arg[8] : sigv : attenuation dans le vol. de véget.
                  arg[9] : Tvol
    """
    if len(arg) != 10:
        print '=== ! Erreur des paramètrees d\entré ! ==='
        return np.nan
    else:
        nbarg = len(arg)
        Ups_est = arg[0]
        vec_kz = arg[1]
        costheta = arg[2]
        Na = arg[3]
        hv = arg[4]    
        Tground = arg[5]
        N = arg[6]
        zg = arg[7]
        sig = arg[8]
        Tvol = arg[9]
                
        vec_gt = X
        Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg)
                 
        V = N*(np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
               np.trace(Ups_est.dot(npl.inv(Ups_mod)))))

        return V   

def monte_carl_estim_dpct_kz(param):
    """ Analyse de monte Carlo de l'estimateur de projection de baseline.
    dans un cadre dual baseline
    
    Les paramètres de la simulation MonteCarlo (P=nombre de real, 
    vec_N Taille d'echantillon ) sont définies dans la fonction
    
    **Entrées** :  
    * *param* : classe de paramètre RVoG
    
    **Sorties**:
	    * *hv,sig,gt1,gt2,gt3* : valeurs estimées des paramètres (Real,Taille_echantillon))
		                         
	    * *meanhv,meansig,meanggti* : moyenne de paramètres estimés (vecteur Taille_echantillon)
                                  
	    * *varhv,varsig,vargti* : var de paramètres estimés (vecteur Taille_echantillon )
    """

    P = 100#nom
    Nb_N=15#nb de taille diff
    vec_N = np.floor((np.logspace(2,5,Nb_N)))
    data_synt=tom.TomoSARDataSet_synth(param.Na,param)
    hv = np.zeros((P,vec_N.size))
    sig = np.zeros((P,vec_N.size))
    gt1= np.zeros((P,vec_N.size))
    gt2= np.zeros((P,vec_N.size))
    gt3= np.zeros((P,vec_N.size))    
    
    for idxN,nb_echant in enumerate(vec_N):
        print '-------------------- N={0}----------'.format(nb_echant)
        for idxP in range(P):
            print 'P={0}'.format(idxP)
            W_k_norm=data_synt.get_W_k_norm_rect(param,int(nb_echant),param.Na)                    
            _,_,_,_,_,_,_,_,_,_,\
            hv_min,sig_min,gt_min =estim_dpct_kz(W_k_norm,param)    
            hv[idxP,idxN] = np.median(hv_min)
            sig[idxP,idxN] = np.median(sig_min)
            
            gt1[idxP,idxN] = 1/gt_min[0]
            gt2[idxP,idxN] = 1/gt_min[1]
            gt3[idxP,idxN] = 1/gt_min[2]
    meanhv=np.mean(hv,axis=0)
    meansig=np.mean(sig,axis=0)
    meangt1=np.mean(gt1,axis=0)
    meangt2=np.mean(gt2,axis=0)
    meangt3=np.mean(gt3,axis=0)
    varhv=np.var(hv,axis=0)
    varsig=np.var(sig,axis=0)
    vargt1=np.var(gt1,axis=0)
    vargt2=np.var(gt2,axis=0)
    vargt3=np.var(gt3,axis=0)
    return hv,sig,gt1,gt2,gt3,\
           meanhv,varhv,meansig,\
           varsig,meangt1,vargt1,\
           meangt2,vargt2,meangt3,\
           vargt3,vec_N,P
       
def monte_carl_estim_ecart_ang_opt(param):
    """ Analyse de monte Carlo de l'estimateur estim_ecart_ang_opt
    
    Les paramètres de la simulation MonteCarlo (P=nombre de real, 
    vec_N Taille d'echantillon ) sont définies dans la fonction
    
    **Entrées** :  
    * *param* : classe de paramètre RVoG
    
    **Sorties**:

    """

    
    P = 50#nb_real
    Nb_N=5#nb de taille diff
    vec_N = np.floor((np.logspace(2,6,Nb_N)))
    data_synt=tom.TomoSARDataSet_synth(param.Na,param)
    Nbase = (param.Na*(param.Na-1))/2
    
    hvJ = np.zeros((P,vec_N.size))
    hvJ2 = np.zeros((P,vec_N.size))
    hvV = np.zeros((P,vec_N.size))
    sigJ = np.zeros((P,vec_N.size))
    sigJ2 = np.zeros((P,vec_N.size))
    sigV = np.zeros((P,vec_N.size))
    vec_gt_J = np.zeros((P,vec_N.size,Nbase))
    vec_gt_J2=np.zeros((P,vec_N.size,Nbase))
    vec_gt_V=np.zeros((P,vec_N.size,Nbase))
    
    for idxN,nb_echant in enumerate(vec_N):
        print '-------------------- N={0}----------'.format(nb_echant)
        for idxP in range(P):            
             print 'P={0}'.format(idxP)

             W_k=data_synt.get_W_k_rect(param,int(nb_echant),param.Na)           
             
             hvJ[idxP,idxN],hvJ2[idxP,idxN],hvV[idxP,idxN],\
             sigJ[idxP,idxN],sigJ2[idxP,idxN],sigV[idxP,idxN],\
             vec_gt_J[idxP,idxN,:],vec_gt_J2[idxP,idxN,:],vec_gt_V[idxP,idxN,:],\
             _,_,_,_,_,_,_,= estim_ecart_ang_opt(W_k,param)                                
             
    #moyenne fction de N
    meanhvJ=np.mean(hvJ,axis=0)
    meanhvJ2=np.mean(hvJ2,axis=0)
    meanhvV=np.mean(sigV,axis=0)
    meansigJ=np.mean(sigJ,axis=0)
    meansigJ2=np.mean(sigJ2,axis=0)
    meansigV=np.mean(sigV,axis=0)
    meangtJ=np.mean(vec_gt_J,axis=0)
    meangtJ2=np.mean(vec_gt_J2,axis=0)
    meangtV=np.mean(vec_gt_V,axis=0)
    
    #var fction de N
    varhvJ=np.var(hvJ,axis=0)
    varhvJ2=np.var(hvJ2,axis=0)
    varhvV=np.var(sigV,axis=0)
    varsigJ=np.var(sigJ,axis=0)
    varsigJ2=np.var(sigJ2,axis=0)
    varsigV=np.var(sigV,axis=0)
    vargtJ=np.var(vec_gt_J,axis=0)
    vargtJ2=np.var(vec_gt_J2,axis=0)
    vargtV=np.var(vec_gt_V,axis=0)

    
    return    hvJ,hvJ2,hvV,\
              sigJ,sigJ2,sigV,\
              vec_gt_J,vec_gt_J2,vec_gt_V,\
              meanhvJ,meanhvJ2,meanhvV,\
              varhvJ,varhvJ2,varhvV,\
              meansigJ,meansigJ2,meansigV,\
              varsigJ,varsigJ2,varsigV,\
              meangtJ,meangtJ2,meangtV,\
              vargtJ,vargtJ2,vargtV,\
              vec_N,P
              
def monte_carl_estim_ecart_ang_opt2(param):
    P = 10#nb_real
    Nb_N=5#nb de taille diff
    vec_N = np.floor((np.logspace(np.log10(5),5,Nb_N)))
    data_synt=tom.TomoSARDataSet_synth(param.Na,param)
    Nbase = (param.Na*(param.Na-1))/2
    
    hvJ = np.zeros((P,vec_N.size))
    hvJ2 = np.zeros((P,vec_N.size))
    hvV = np.zeros((P,vec_N.size))
    sigJ = np.zeros((P,vec_N.size))
    sigJ2 = np.zeros((P,vec_N.size))
    sigV = np.zeros((P,vec_N.size))
    vec_gt_J = np.zeros((P,vec_N.size,Nbase))
    vec_gt_J2=np.zeros((P,vec_N.size,Nbase))
    vec_gt_V=np.zeros((P,vec_N.size,Nbase))
    
    for idxN,nb_echant in enumerate(vec_N):
        print '-------------------- N={0}----------'.format(nb_echant)
        for idxP in range(P):            
             print 'P={0}'.format(idxP)

             W_k=data_synt.get_W_k_rect(param,int(nb_echant),param.Na)           
             
             hvJ[idxP,idxN],hvJ2[idxP,idxN],hvV[idxP,idxN],\
             sigJ[idxP,idxN],sigJ2[idxP,idxN],sigV[idxP,idxN],\
             vec_gt_J[idxP,idxN,:],vec_gt_J2[idxP,idxN,:],vec_gt_V[idxP,idxN,:],\
             _,_,_,_,_,_,_,= estim_ecart_ang_opt2(W_k,param)                                
             
    #moyenne fction de N
    meanhvJ=np.mean(hvJ,axis=0)
    meanhvJ2=np.mean(hvJ2,axis=0)
    meanhvV=np.mean(sigV,axis=0)
    meansigJ=np.mean(sigJ,axis=0)
    meansigJ2=np.mean(sigJ2,axis=0)
    meansigV=np.mean(sigV,axis=0)
    meangtJ=np.mean(vec_gt_J,axis=0)
    meangtJ2=np.mean(vec_gt_J2,axis=0)
    meangtV=np.mean(vec_gt_V,axis=0)
    
    #var fction de N
    varhvJ=np.var(hvJ,axis=0)
    varhvJ2=np.var(hvJ2,axis=0)
    varhvV=np.var(sigV,axis=0)
    varsigJ=np.var(sigJ,axis=0)
    varsigJ2=np.var(sigJ2,axis=0)
    varsigV=np.var(sigV,axis=0)
    vargtJ=np.var(vec_gt_J,axis=0)
    vargtJ2=np.var(vec_gt_J2,axis=0)
    vargtV=np.var(vec_gt_V,axis=0)

    return    hvJ,hvJ2,hvV,\
              sigJ,sigJ2,sigV,\
              vec_gt_J,vec_gt_J2,vec_gt_V,\
              meanhvJ,meanhvJ2,meanhvV,\
              varhvJ,varhvJ2,varhvV,\
              meansigJ,meansigJ2,meansigV,\
              varsigJ,varsigJ2,varsigV,\
              meangtJ,meangtJ2,meangtV,\
              vargtJ,vargtJ2,vargtV,\
              vec_N,P
              
def mont_carl_estim_ang_scal(param,param_simu):
    
    P = param_simu['P']#nb_real
    Nb_N=param_simu['NbN'] #nb de taille diff
    Nmin=param_simu['Nmin']
    Nmax=param_simu['Nmax']
    vec_N = np.floor((np.logspace(Nmin,Nmax,Nb_N)))
    data_synt=tom.TomoSARDataSet_synth(param)
    #Nbase = (param.Na*(param.Na-1))/2
    
    mat_hvJscal = np.zeros((P,vec_N.size))
    mat_sigvJscal = np.zeros((P,vec_N.size))
    for idxN,nb_echant in enumerate(vec_N):        
        print '-------------------- N={0}----------'.format(nb_echant)
        for idxP in range(P):            
             print 'P={0}'.format(idxP)
             W_k=data_synt.get_W_k_rect(param,int(nb_echant))                        
             mat_hvJscal[idxP,idxN],mat_sigvJscal[idxP,idxN],_ = \
             estim_ecart_ang_scal(W_k,param)
             
    vec_meanhvJscal = np.mean(mat_hvJscal,axis=0)
    vec_meansigvJscal = np.mean(mat_hvJscal,axis=0)
    
    vec_varhvJscal= np.var(mat_hvJscal,axis=0)
    vec_varsigvJscal= np.var(mat_sigvJscal,axis=0)
    
    
    
    return mat_hvJscal,mat_sigvJscal,\
           vec_meanhvJscal,vec_meansigvJscal,\
           vec_varhvJscal,vec_varsigvJscal
           
def monte_carl_estim_mlogV_Tv_Tg_zg_known(param,param_simu):
    """Analyse Monte Carlo des estimateur pour le critère mlogV_Tv_Tg_zg_known"""
    P = param_simu['P']#nb_real
    Nb_N=param_simu['NbN'] #nb de taille diff
    Nmin=param_simu['Nmin']
    Nmax=param_simu['Nmax']
    U0=param_simu['U0']#initial guess pour mlogV_Tv_Tg_zg_known 
    
    vec_N = np.floor((np.logspace(np.log10(Nmin),np.log10(Nmax),Nb_N)))
        
    data_synt=tom.TomoSARDataSet_synth(param)
    Nbase = mb.get_Nb_from_Na(param.Na)#Nbre de baseline i.e de deco temporelles 
    mat_hvV = np.zeros((P,vec_N.size))
    mat_sigvV = np.zeros((P,vec_N.size))
    mat_gtV = np.zeros((P,vec_N.size,Nbase))
    for idxN,nb_echant in enumerate(vec_N):        
        print '-------------------- N={0}----------'.format(nb_echant)
        param.N = nb_echant
        for idxP in range(P):            
            print 'P={0}'.format(idxP)         
            W = data_synt.get_W_k_rect(param,int(nb_echant))

            mat_hvV[idxP,idxN],mat_sigvV[idxP,idxN],\
            mat_gtV[idxP,idxN,:] = estim_mlogV_Tg_Tv_zg_known(W,param,U0=U0)
            print mat_hvV[idxP,idxN],mat_sigvV[idxP,idxN],mat_gtV[idxP,idxN,:]
            
    #moyenne fction de N   
    mean_hvV=np.mean(mat_hvV,axis=0)
    mean_sigV=np.mean(mat_sigvV,axis=0)    
    mean_gtV=np.mean(mat_gtV,axis=0)
    
    #var fction de N    
    var_hvV=np.var(mat_hvV,axis=0)
    var_sigV=np.var(mat_sigvV,axis=0)
    var_gtV=np.var(mat_gtV,axis=0)
        
    return mat_hvV,mat_sigvV,mat_gtV,\
           mean_hvV,mean_sigV,mean_gtV,\
           var_hvV,var_sigV,var_gtV,\
           vec_N,P
    
    
def monte_carl_estim_mlogV_zg_known_et_J_scal(param,param_simu):
    """Analyse Monte Carlo des estimateur pour les critères J, J2
    et de vraissemblance.
    
    Optimisation des critères J_scal et mlogV_zg_known """
    
    P = param_simu['P']#nb_real
    Nb_N=param_simu['NbN'] #nb de taille diff
    Nmin=param_simu['Nmin']
    Nmax=param_simu['Nmax']
    UU0=param_simu['UU0']#initial guess pour mlogV_zg_known 
    U0=param_simu['U0']#initial guess pour J_scal

    vec_N = np.floor((np.logspace(np.log10(Nmin),np.log10(Nmax),Nb_N)))
        
    data_synt=tom.TomoSARDataSet_synth(param)
    Nbase = mb.get_Nb_from_Na(param.Na)#Nbre de baseline i.e de deco temporelles 
    
    mat_hvJ = np.zeros((P,vec_N.size))
    mat_sigvJ = np.zeros((P,vec_N.size))
    mat_gtJ = np.zeros((P,vec_N.size,Nbase))
    mat_bJ = np.zeros((P,vec_N.size))
    
    mat_hvJ2 = np.zeros((P,vec_N.size))
    mat_sigvJ2 = np.zeros((P,vec_N.size))    
    mat_gtJ2 = np.zeros((P,vec_N.size,Nbase))
    mat_bJ2 = np.zeros((P,vec_N.size))
    
    mat_hvV = np.zeros((P,vec_N.size))
    mat_sigvV = np.zeros((P,vec_N.size))
    mat_gtV = np.zeros((P,vec_N.size,Nbase))
    mat_reu_V = np.zeros((P,vec_N.size)) #contient 1 si l'opti a convergé
    
    for idxN,nb_echant in enumerate(vec_N):        
        print '-------------------- N={0}----------'.format(nb_echant)
        param.N = nb_echant
        for idxP in range(P):            
            
            W = data_synt.get_W_k_rect(param,int(nb_echant))
            k1 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))            
            k2 = 0.5*(np.random.randn(3,1)+1j*np.random.randn(3,1))
            Tvoln = param.T_vol + k1.dot(k1.T.conj())                       
            Tgroundn = param.T_ground + k2.dot(k2.T.conj())                       
    
            UU0 = np.concatenate((mb.get_vecTm_from_Tm(Tvoln),
                                 mb.get_vecTm_from_Tm(Tgroundn),
                                 np.array([param.h_v+dX[18]]),
                                 np.array([param.sigma_v+dX[19]]),
                                 0.8*np.ones(3)))    
            
            mat_hvJ[idxP,idxN],mat_sigvJ[idxP,idxN],\
            mat_gtJ[idxP,idxN,:],mat_bJ[idxP,idxN] = estim_ecart_ang_scal(W,param,critere='J',zg_connu=param.z_g,U0=U0)
            
            mat_hvJ2[idxP,idxN],mat_sigvJ2[idxP,idxN],\
            mat_gtJ2[idxP,idxN,:],mat_bJ2[idxP,idxN] = estim_ecart_ang_scal(W,param,critere='J2',zg_connu=param.z_g,U0=U0)
            
            _,_,mat_hvV[idxP,idxN],mat_sigvV[idxP,idxN],\
            mat_gtV[idxP,idxN,:],\
            mat_reu_V[idxP,idxN] = estim_mlogV_zg_known(W,param,U0=UU0)
            print mat_hvV[idxP,idxN],mat_sigvV[idxP,idxN],mat_gtV[idxP,idxN,:]
            
    #moyenne fction de N
    mean_hvJ = np.mean(mat_hvJ,axis=0)
    mean_sigJ = np.mean(mat_sigvJ,axis=0)
    mean_gtJ = np.mean(mat_gtJ,axis=0)
    mean_bJ = np.mean(mat_bJ,axis=0)
    
    mean_hvJ2 = np.mean(mat_hvJ2,axis=0)
    mean_sigJ2 = np.mean(mat_sigvJ2,axis=0)    
    mean_gtJ2 = np.mean(mat_gtJ2,axis=0)
    mean_bJ2 = np.mean(mat_bJ,axis=0)
    
    mean_hvV = np.mean(mat_hvV,axis=0)
    mean_sigV = np.mean(mat_sigvV,axis=0)    
    mean_gtV = np.mean(mat_gtV,axis=0)
    
    #var fction de N
    var_hvJ = np.var(mat_hvJ,axis=0)
    var_sigJ = np.var(mat_sigvJ,axis=0)
    var_gtJ = np.var(mat_gtJ,axis=0)
    var_bJ = np.var(mat_bJ,axis=0)
    
    var_hvJ2 = np.var(mat_hvJ2,axis=0)
    var_sigJ2 = np.var(mat_sigvJ2,axis=0)    
    var_gtJ2 = np.var(mat_gtJ2,axis=0)
    var_bJ2 = np.var(mat_bJ2,axis=0)
    
    var_hvV = np.var(mat_hvV,axis=0)
    var_sigV = np.var(mat_sigvV,axis=0)
    var_gtV = np.var(mat_gtV,axis=0)
        
    return mat_hvJ,mat_sigvJ,mat_gtJ,mat_bJ,\
           mat_hvJ2,mat_sigvJ2,mat_gtJ2,mat_bJ2,\
           mat_hvV,mat_sigvV,mat_gtV,\
           mean_hvJ,mean_sigJ,mean_gtJ,mean_bJ,\
           mean_hvJ2,mean_sigJ2,mean_gtJ2,mean_bJ2,\
           mean_hvV,mean_sigV,mean_gtV,\
           var_hvJ,var_sigJ,var_gtJ,var_bJ,\
           var_hvJ2,var_sigJ2,var_gtJ2,var_bJ2,\
           var_hvV,var_sigV,var_gtV,\
           vec_N,P,mat_reu_V
    

    
def EQM(Ups_est,hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground):    
    """Calcul de l'erreur quadratique entre la matruc *Ups_est*
    et la matrice Ups reconstruite à partir du modèle RVoG
    
    **Entrée**
        * *Ups_est* : matrice de covariance (base lexicographique)
        * *hv* : hauteur de végétation
        * *sig* : sigmav (attenuation dans le volume)
        * *vec_gt* :ens des décohe temp (ordre : gt12, gt13, gt23 ...)
        * *vec_kz* : ens des nbre d'onde (ordre : kz12, kz13, kz23 ...)
        * *costheta* : cos de l'angle incidence du radar
        * *Na* : nombre d'antennes1
        * *Tvol* : réponse polar du volume 
        * *Tground* : réponse polar du sol 
    
    **Sortie**
        * *EQM* : erreur quadratique"""
        
    Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground)
    diff = Ups_est-Ups_mod
    return np.trace(diff.dot(diff.T.conj()))

def ressemblance(Ups_est,N,hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground):
    """Calcul de la -logVraissmeblance entre *Ups_est* et Ups obtenue à l'aide 
    des paramètres RVoG
    
    **Entrée**
        * *Ups_est* : matrice de covariance (base lexicographique)
        * *N* : taille de l'echantillon (calul de Ups_est)
        * *hv* : hauteur de végétation
        * *sig* : sigmav (attenuation dans le volume)
        * *vec_gt* :ens des décohe temp (ordre : gt12, gt13, gt23 ...)
        * *vec_kz* : ens des nbre d'onde (ordre : kz12, kz13, kz23 ...)
        * *costheta* : cos de l'angle incidence du radar
        * *Na* : nombre d'antennes1
        * *Tvol* : réponse polar du volume 
        * *Tground* : réponse polar du sol 
    
    **Sortie**
        * *vraissemblance* """
    
    Ups_mod = Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground)
    return vraissemb(Ups_est,Ups_mod,N)
    
def vraissemb(Ups_est,Ups_mod,N):
    """Renvoi la -logVraissemblance a partir des matrices estimé 
    *Ups_est* et issue du modèle RVoG *Ups_mod*.
    
    **Entrées**
        * *Ups_est* : matrice de covariance (base lexicographique)
        * *Ups_mod* : matrice de covariance obtenueà partir des param RVoG
        * *N*       : taille de l'echantillon (estimation de Ups)

    **Sortie**
        * *-logVraissemblance* """
    
    Na = int(Ups_mod.shape[0]/3)    
    if np.real(npl.det(Ups_mod))<0:
        print 'vraissemb: det(Ups_mod) nég! -> return nan'
        return np.nan
    if npl.cond(Ups_mod)>10**6:
        print 'vraissemb: Ups_mod mal conditionné: Cond={0}'\
                .format(npl.cond(Ups_mod))
        return np.nan
    
    V = np.real(np.log(np.pi**(3*Na)*npl.det(Ups_mod))+\
        N*np.trace(Ups_est.dot(npl.inv(Ups_mod))))
        #mathmtqt tr(AB)€R si A,B hermi (d'où la partie reelle)
    return V

def Ups_reconstr(hv,sig,vec_gt,vec_kz,costheta,Na,Tvol,Tground,zg):    
    """Reconstruit la matrice Ups (modèle RVoG) à partir des
    paramètres du modèle.
    
    **Entrée**    
        * *hv* : hauteur de végétation 
        * *sig* : attenuation des ondes    
        * *vec_gt* : ens des décorrelation temporelle sur chaque baseline (ordre : gt12, gt13, gt23 ...)
        * *vec_kz* : ens des nbre d'onde (ordre : kz12, kz13, kz23 ...)
        * *costheta* : cos de l'angle incidence du radar
        * *Na* : nombre d'antennes1
        * *Tvol* : réponse polar du volume 
        * *Tground* : réponse polar du sol 
        * *zg* : altitude du sol            
        
    **Sortie**
        * *Ups_recon* : matrice Ups reconstruite """
    
    Ups_recon = np.zeros((3*Na,3*Na),dtype='float')
    R_sol=np.ones((Na,Na),dtype='complex')
    R_volume=np.ones((Na,Na),dtype='complex')
    p = 0
    alpha = 2*sig/costheta
    a = np.exp(-alpha*hv)
    if alpha != 0:
        I1=(1-a)/alpha
    else:
        #Prolongement par continuite
        I1 = hv
        
    for i in range(Na-1):
        for j in range(i+1,Na):                      
            exp_ikzij_zg = np.exp(1j*vec_kz[p]*zg)                   
            R_volume[i,j] = exp_ikzij_zg*gammavgt(hv,sig,costheta,vec_kz[p],float(vec_gt[p]))
            R_volume[j,i]=np.conj(R_volume[i,j])
            R_sol[i,j]=exp_ikzij_zg
            R_sol[j,i]=np.conj(R_sol[i,j])            
            p = p+1
            
    Ups_recon = np.kron(R_sol,a*Tground)+np.kron(R_volume,I1*Tvol)    
    return Ups_recon

def Ups_reconstr2(hv,sig,vec_gt,vec_kz,
                  costheta,Na,I1Tvol,aTground,zg):                      
    """Renconstruit la matrice Ups (modèle RVoG) à partir des
    paramètres du modèle RVoG.
    
    Dans cette version on prend en compte les matrices Cv=I1Tvol et Cg=aTground obtenue
    directement en sortie de SKP : W = Cv o Rv + Cg o Rg.
    
    **Entrées**
        * *hv* : hauteur de végétation 
        * *sig* : attenuation des ondes    
        * *vec_gt* : ens des décorrelation temporelle sur chaque baseline (ordre : gt12, gt13, gt23 ...)
        * *vec_kz* : ens des nbre d'onde (ordre : kz12, kz13, kz23 ...)
        * *costheta* : cos de l'angle incidence du radar
        * *Na* : nombre d'antennes
        * *I1Tvol* : I1*réponse polar du volume (I1 var intermediare)
        * *aTground* : a*réponse polar du sol (a var intermediare RVog)            
        * *zg* : altitude du sol 
        
    **Sortie**
        * *Ups_recon* : matrice Ups reconstruite
    """    
    
    Ups_recon2 = np.zeros((3*Na,3*Na),dtype='complex')
    R_sol=np.ones((3,3),dtype='complex')
    R_volume=np.ones((Na,Na),dtype='complex')
    p=0
    for i in range(Na-1):
        for j in range(i+1,Na):                     
            exp_ikzij_zg = np.exp(1j*vec_kz[p]*zg)                   
            R_volume[i,j] = exp_ikzij_zg*gammavgt(hv,sig,costheta,vec_kz[p],float(vec_gt[p]))
            R_volume[j,i] = np.conj(R_volume[i,j])
            R_sol[i,j]=exp_ikzij_zg
            R_sol[j,i]=np.conj(R_sol[i,j])
            p = p+1
            
    Ups_recon2 = np.kron(R_sol,aTground)+np.kron(R_volume,I1Tvol)          
    return Ups_recon2
    
def crit_ang(vec_gm,hv,sigv,costheta,vec_kz):
    """Calcul du critère J (initialement nommé critère angulaire)
    
    **Entrées**    
        * *vec_gm* : cohérence du volume estimées (ordre : gv12, gv13, gv23 ...)
        * *hv* :hauteur de végétation
        * *costheta* : cos de l'angle d'incidence
        * *vec_kz* : ens des nbre d'onde (ordre : kz12, kz13, kz23 ...)
        
    **Sortie**
        * *crit* : critère J"""    
    Nb_gma = vec_gm.size
    crit=0
    for i in range(Nb_gma):
        gvi = gammav(hv,sigv,costheta,vec_kz[i])
        crit = crit + np.abs(vec_gm[i]- gvi/np.abs(gvi)*np.abs(vec_gm[i]))**2
    return crit

def crit_ang2(vec_gm,hv,sigv,costheta,vec_kz):
    """Calcul du critère J2 (initialement nommé critère angulaire)
    
    **Entrées**    
        * *vec_gm* : cohérence du volume estimées (ordre : gv12, gv13, gv23 ...)
        * *hv* :hauteur de végétation
        * *costheta* : cos de l'angle d'incidence
        * *vec_kz* : ens des nbre d'onde (ordre : kz12, kz13, kz23 ...)
        
    **Sortie**
        * *crit* : critère J2"""    
        
    Nb_gma = vec_gm.size
    crit=0
    for i in range(Nb_gma):
        gvi = gammav(hv,sigv,costheta,vec_kz[i])
        crit = crit + np.abs(vec_gm[i]/np.abs(vec_gm[i])- gvi/np.abs(gvi))**2
    return crit
    
def load_nv_phiv(path = '/data2/pascale/'):
    """Chargement lookup table (nv,phiv)"""
    
    try:
        ima_nv = pym.Image(path+'img_nv3',load=True)
        ima_phiv = pym.Image(path+'img_phiv3',load=True)
    except:
        ima_nv,ima_phiv =load_nv_phiv()
        
    return ima_nv,ima_phiv   
          

    
def launch_ex():
    Na = 3
    Np = 3
    A = 0.95
    E = 200
    k_z = [0.1,0.15]
    #MB    
    param_MB = lp.load_param('DB_0')
    param_MB = mb.rvog_reduction(param_MB,A,E)
    param_MB.k_z=k_z
    param_MB.Na=len(param_MB.k_z)+1
    param_MB.theta=45*np.pi/180
    param_MB.sigma_v=0.0345
    param_MB.h_v=10
    if param_MB.h_v > np.min(2*np.pi/np.array(k_z)):print 'Attention h_v > Hamb'
    param_MB.z_g = 0
    param_MB.gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt())
     
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
    b_vrai = tom.b_true(param_MB)
    _,Rg,Rv,Cg,Cv=tom.value_R_C(R_t,C_t,a,b)    
    
    vec_gt = np.array([param_MB.gammat[0,1],param_MB.gammat[0,2],param_MB.gammat[1,2]])
    estim_ecart_ang_opt2   
    
    hvJ,hvJ2,hvV,\
    sigJ,sigJ2,sigMV,\
    vec_gt_J,vec_gt_J2,vec_gt_MV,\
    X_minJ,X_minJ2,X_minV,\
    minJ,minJ2,minV,\
    vec_b = estim_ecart_ang_opt2(W_k,param_MB,zg_connu=param_MB.z_g)
 
    b_vrai = tom.b_true(param_MB) 
    b_vrai_num = vec_b[np.argmin(np.abs(vec_b-b_vrai))]

    
    plt.figure()
    plt.plot(vec_b,minJ,'b.-',label='J')
    plt.plot(vec_b,minJ2,'r.-',label='J2')
    plt.plot(vec_b,(minV),'g.-',label='-logV-med(-logV)')
    plt.axvline(x=b_vrai_num,ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3)
    plt.title('Variation critere en fonction de b')
    plt.grid()
    plt.legend()
    
    plt.figure()
    plt.semilogy(vec_b,minJ,'b.-',label='J')
    plt.semilogy(vec_b,minJ2,'r.-',label='J2')
    plt.semilogy(vec_b,minV,'g.-',label='-logV')
    plt.axvline(x=b_vrai_num,ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3)
    plt.title('Variation critere en fonction de b (log)')
    plt.grid()
    plt.legend()
    
    plt.figure()
    plt.plot(vec_b,(minJ-np.min(minJ)),'b.-',label='J')
    plt.plot(vec_b,(minJ2-np.min(minJ2)),'r.-',label='J2')
    plt.plot(vec_b,(minV-np.min(minV)),'g.-',label='-logV')
    plt.axvline(x=b_vrai_num,ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3)
    plt.title('Variation critere en fonction de b (soustraction du min)')
    plt.grid()
    plt.legend()
        
    plt.figure()
    plt.semilogy(vec_b,(minJ-np.min(minJ)),'b.-',label='J')
    plt.semilogy(vec_b,(minJ2-np.min(minJ2)),'r.-',label='J2')
    plt.semilogy(vec_b,(minV-np.min(minV)),'g.-',label='-logV')
    plt.axvline(x=b_vrai_num,ymin=0,ymax=1,alpha=.7,linestyle='--',color='k',
                lw=3)
    plt.title('Variation critere en fonction de b (log/soustraction du min)')
    plt.grid()
    plt.legend()
    
    

        
if __name__ == "__main__":
    launch_ex()