# -*- coding: utf-8 -*-
"""Classe param_rvog et routines associées"""
from __future__ import division
import numpy as np
import sys
import copy as cp
import cmath as cm
import numpy.linalg as npl
import matplotlib.pyplot as plt
import basic_lib as bl
import load_param as lp
import pdb
plt.ion()

class param_rvog:
    def __init__(self,N,k_z,theta,T_vol,T_ground,h_v,z_g,sigma_v,mat_gamma_t):
        """
        Attributs:
        
        * N : taille de l\'echantillon
        * Na : Nombre d\'antennes
        * k_z : *list* liste contenant les kz sous la forme kz12,kz13,..,kz1Na          
        * theta : angle d\'incidence 
        * T_vol : rep. polar. du volume
        * T_ground : rep. polar. du sol
        * h_v : hauteur de vegetation
        * sigma_v : attenuation dans le volume
        * gammat : *array* Tableau de taille NaxNa contenant gt_ij, décoherence 
                  temporelle entre les antennes i et j.
                  
        """
        self.N = N
        self.Na = len(k_z)+1
        self.k_z = k_z        
        self.theta = theta
        self.T_vol = T_vol
        self.T_ground = T_ground
        self.h_v = h_v
        self.z_g = z_g
        self.sigma_v = sigma_v
        self.gammat = mat_gamma_t #seul les i<j sont utilisés
        self.param_format()
        
    def param_format(self):
        """Converti les attributs de param_rvog dans les formats adaptés."""
        self.N = int(self.N)                
        self.Na = int(self.Na)
        try:
            self.k_z=[float(X) for X in self.k_z]
            self.theta=float(self.theta)
            self.h_v=float(self.h_v)    
            self.z_g=float(self.z_g)
            self.sigma_v=float(self.sigma_v)
            self.gammat=self.gammat.astype(float)    
            
        except TypeError as err:
            print 'kz,theta,hv,zg,sigv,gamamat ne sont pas cpx\n{0}'.format(err)
                    
        self.T_vol=self.T_vol.astype(complex)
        self.T_ground=self.T_ground.astype(complex)        
        
    def display(self):
        """ Affiche les valeurs des paramètres."""
        print 'N {0} k_z {1} theta {2}'.format(str(self.N),str(self.k_z),str(self.theta))\
                +'\n'+ \
               'h_v {0} z_g {1} sigma_v {2}'.format(str(self.h_v),str(self.z_g),str(self.sigma_v)) \
                      
        print '\ngammat'
        print '\n'.join(' '.join(str(cell) for cell in row) for row in self.gammat)
        print '\nT_vol '
        print '\n'.join(' '.join(str(cell) for cell in row) for row in self.T_vol)
        print '\nT_ground '
        print '\n'.join(' '.join(str(cell) for cell in row) for row in self.T_ground)
        
    def get_I1(self):
        """ Calcul de I1 (variable intermediaire du modèle RVoG)."""
        a=np.exp((-2*self.sigma_v*self.h_v)/(np.cos(self.theta)))
        alpha=2*self.sigma_v/np.cos(self.theta)    
        I1=(1-a)/alpha
        return I1
        
    def get_I2(self,i=0,j=1):
        """Calul de I2 pour la baseline entre les antennes ``i`` et ``j``"""        
        
        a=np.exp((-2*self.sigma_v*self.h_v)/(np.cos(self.theta)))
        alpha=2*self.sigma_v/np.cos(self.theta)            
        I2 = (np.exp(1j*self.get_k_z(i,j,self.Na)*self.h_v)-a)/ \
                    (1j*self.get_k_z(i,j,self.Na)+alpha)
        return I2
    
    def get_gamma_v(self,i=0,j=1):
        """Renvoie la cohérence interfero du volume seul gamma_v
            entre les antennes ``i`` et ``j``"""
        return self.get_I2(i,j)/self.get_I1()
        
    def get_gamma_v_gt(self,i=0,j=1):       
        """Renvoie gamma_v*gammat(i,j)"""
        
        return self.get_I2(i,j)/self.get_I1()*self.get_gamma_t(i,j)
        
    def get_gamma_t(self,i=0,j=1):
        """Renvoie la décoherence temporelle entre les antennes ``i`` et ``j``"""
        return self.gammat[i,j]
        
    def get_a(self):       
        """ Cacul de a (variable intermediaire du modèle RVoG) """
        
        a=np.exp((-2*self.sigma_v*self.h_v)/(np.cos(self.theta)))
        return a
        
    def get_Rv(self):
        """ Cacul de la matrice de structure du volume (cf SKP/Tebaldini)"""
        
        R_v =np.eye(self.Na,dtype='complex')
        #parcours des i<j
        for i in range(self.Na-1):
            for j in range(i+1,self.Na):
                exp_Phig = np.exp(1j*self.get_k_z(i,j,self.Na)*self.z_g)
                R_v[i,j] = exp_Phig*self.get_I2(i,j)*self.get_gamma_t(i,j)/self.get_I1()
                R_v[j,i] = np.conj(R_v[i,j])        
        return R_v
        
    def get_Rv_noground(self):
        """Renvoie Rv sans le terme de la phase du sol """
        
        R_v =np.zeros((self.Na,self.Na),dtype='complex')
        #parcours des i<j
        for i in range(self.Na-1):
            for j in range(i+1,self.Na):                
                R_v[i,j] = self.get_I2(i,j)*self.get_gamma_t(i,j)/self.get_I1()                
        R_v = (R_v.T.conj()+R_v)+np.eye(self.Na)
        return R_v
        
    def get_Rg(self):
        """ Cacul de la matrice de structure du sol (cf SKP/Tebaldini)"""
        
        Rg = np.zeros((self.Na,self.Na),dtype='complex')
        #parcours des i<j
        for i in range(self.Na-1):
            for j in range(i+1,self.Na):
                exp_Phig = np.exp(1j*self.get_k_z(i,j,self.Na)*self.z_g)
                Rg[i,j] = exp_Phig
        Rg = (Rg.T.conj()+Rg)+np.eye(self.Na)            
        return Rg
        
    def get_h_amb(self,idx_b=0):
        """ Renvoie la hauteur d'ambiguité pour la ``idx_b`` eme baseline"""
        
        return 2*np.pi/self.k_z[idx_b] #En sb il n'y a qu'un k_z
        
    def get_hamb_min(self):
        """Renvoie la hauteur d'ambiguite minimale parmi toutes les baselines"""
        
        Nb = get_Nb_from_Na(self.Na)
        vec_hamb = np.zeros(Nb)
        vec_kz = self.get_kzlist()
        vec_hamb = 2.*np.pi/vec_kz
        return np.min(vec_hamb)
        
    def get_k_z(self,ant1,ant2,Na):
        """ Renvoie le k_z entre l'antenne ``ant1`` et l'antenne ``ant2``"""
        k_z = self.k_z        

        if ant1 < ant2:
            signe = 1
        elif ant1 > ant2:
            signe = -1
        else:
             print 'Erreur: ant1=ant2!'
             return 0                           
        if ant1 == 0 or ant2 == 0:
            #le k_z est de la forme k_z_1n 
            #donc pas d'operation de chasles necessaires
            k_z_ant1_ant2 = signe*k_z[np.max([ant1,ant2])-1]
        else:            
            k_z_ant1_ant2 = signe*(-k_z[ant1-1]+k_z[ant2-1])                       
        return k_z_ant1_ant2 
    
    def get_kzlist(self):
        """Renvoie la liste de kz suivant l'ordre i<j pour i€[1,Na-1))]
        
        ex pour Na=3 kzlist=kz12 kz13 kz23"""
        
        Na=self.Na
        vec_kz=np.zeros(np.floor(Na*(Na-1)/2),dtype='float')
        p=0
        for i in range(Na-1):
            for j in range(i+1,Na):
                vec_kz[p]=self.get_k_z(i,j,Na)                
                p +=1
        return vec_kz        
        
    def get_Om(self,i,j):
        """Renvoie la rep. interfero. entre les antennes ``i`` et ``j`` """
        
        Ups=self.get_upsilon()
        if i==j :
            print 'Warning : les rep interfero verifient i !=j'
            return bl.nans((3,3))
            
        return Ups[3*i:3*(i+1),3*j:3*(j+1)]
        
    def get_sigma_v_dBm(self):
        """Renvoie sigma_v en dB/m"""
        
        return self.sigma_v*20/np.log(10)        
        
    def get_gtlist(self):
        
        """Renvoie la liste des gammat_ij suivant l'ordre i<j pour i€[1,Na-1))]
        ex pour Na=3 kzlist =gt12 gt13 gt23"""
        
        Na=int(self.Na)
        Nb = int(get_Nb_from_Na(Na))
        vec_gt=np.zeros(Nb,dtype='float')
        p=0
        for i in range(Na-1):
            for j in range(i+1,Na):
                vec_gt[p]=self.gammat[i,j]
                p +=1
        return vec_gt
        
    def get_invariant(self,idx_b=0):
        """        
        Renvoi les 4 paramètres invariants du modèle RVoG décrit par 
        le vecteur de paramètre param_RVoG: A,E,h_v,lambda2
        cf "Invariant Contrast Parameters of PolInSARHomogenous RVoG Model" P.Réfrégie et Al
        """        
        
        T_vol=self.T_vol
        T_ground=self.T_ground
        #valeurs propres de T_vol^-1*T_ground dans l'ordre décroissant
        vec_lmbda,_ =npl.eig(npl.inv(T_vol).dot(T_ground))
        vec_lmbda= sorted(vec_lmbda,reverse=1)
        A = (vec_lmbda[0]-vec_lmbda[2])/(vec_lmbda[0]+vec_lmbda[2])
        E = np.sum(vec_lmbda)
        X = (vec_lmbda[1]-vec_lmbda[2])/(vec_lmbda[0]-vec_lmbda[2])
        return A,E,vec_lmbda[1],self.h_v,X
   
    def get_upsilon(self):
        """
        Retourne la matrice de covariance théorique upsilon 3Nax3Na du modèle RVog.
        """        
        
        Na = self.Na
        theta = self.theta
        T_vol = self.T_vol
        T_ground = self.T_ground
        h_v = self.h_v
        z_g = self.z_g
        sigma_v = self.sigma_v
        if Na != len(self.k_z)+1:
            print 'Erreur Na != nbre de kz_1n'
        #Construction matrice de covariance Upsilon
        Ups=bl.nans([3*Na,3*Na],dtype=np.complex)
        T=bl.nans([3,3],dtype=np.complex)
        Om_ij=np.zeros((3,3),dtype=np.complex)        
        
        alpha=2*sigma_v/np.cos(theta) 
        a=np.exp(-alpha*h_v)           
        I1=(1-a)/alpha
        T=I1*T_vol+a*T_ground
                
        #Dupplication Na fois de la matrice T         
        Ups = np.kron(np.eye(Na)+0*1j,T)
        for i in range(Na-1):           
                for j in range(i+1,Na):                    
                    #parcours les i<j                   
                    I2 = (np.exp(1j*self.get_k_z(i,j,Na)*h_v)-a)/ \
                    (1j*self.get_k_z(i,j,Na)+alpha)
                    exp_Phig = np.exp(1j*self.get_k_z(i,j,Na)*z_g)
                    Om_ij= exp_Phig*(I2*T_vol+a*T_ground)
                    
                    Ups[3*(i):3*(i+1),3*(j):3*(j+1)] = Om_ij
                    Ups[3*(j):3*(j+1),3*(i):3*(i+1)] = Om_ij.T.conj()
      
        return Ups
    
    def get_upsilon_gt(self):
        """
        Retourne la marice de covariance théorique upsilon 3Nax3Na du modèle RVog
        avec décohérence temporelle.
        """        
        
        Na = self.Na
        k_z = self.k_z #vecteur
        theta = self.theta
        T_vol = self.T_vol
        T_ground = self.T_ground
        h_v = self.h_v
        z_g = self.z_g
        sigma_v = self.sigma_v
                
        #Construction matrice de covariance Upsilon
        Ups=bl.nans([3*Na,3*Na],dtype=np.complex)
        T=bl.nans([3,3],dtype=np.complex)
        Om=np.ndarray((Na,Na,3,3),dtype=np.complex)        
        
        a=self.get_a()
        alpha=2*sigma_v/np.cos(theta)    
        I1=(1-a)/alpha
        T=I1*T_vol+a*T_ground        
        
        #Dupplication Na fois de la matrice T         
        Ups = np.kron(np.eye(Na)+0*1j,T)
        
        for i in range(Na-1):           
                for j in range(i+1,Na):                    
                    #parcours les i<j
                    #print 'in get_upsilon' (i,j)
                    I2gt = self.get_I2(i,j)*self.get_gamma_t(i,j)
                    exp_iPhig = np.exp(1j*self.get_k_z(i,j,Na)*z_g)
                    Om[i,j][:,:] = exp_iPhig*(I2gt*T_vol+a*T_ground)

                    Ups[3*i:3*(i+1),3*j:3*(j+1)] = Om[i,j][:,:]
                    Ups[3*j:3*(j+1),3*i:3*(i+1)] = Om[i,j][:,:].T.conj()        
        return Ups
        
    def get_fisher(self):
        """Retourne la matrice de Fisher en fonction du jeu de paramètres param
        
        La matrice de Fisher est exprimé dans la base 
        (Tvol, Tground, h_v, z_g,sigma_v,{gammat}_ij (i<j))
        ex en dual baseline eta=(Tvol, Tground, h_v, z_g,sigma_v,gammat12,gammat13,gammat12)
        """
        
        Na = self.Na
        N = self.N
        k_z = self.k_z
        theta = self.theta
        T_vol = self.T_vol
        T_ground = self.T_ground
        h_v = self.h_v
        z_g = self.z_g
        sigma_v = self.sigma_v
        #var intermediaire
        alpha=2*sigma_v/np.cos(theta)
        a=np.exp(-alpha*h_v)        
        I1=(1-a)/alpha
        
        Nbase = get_Nb_from_Na(self.Na)#Nbre de baseline possibles pour Na acquis
        Nb_inc = 21+Nbase#Nbre d'inconnus
        Ups = self.get_upsilon_gt()#version avec deco temporelle
        #Cacul des dérivés de Upsilon 
        #derive de Ups : liste des derivées de Ups par rapport aux 21+Ngt paramètres
        # eta = (Tvol,Tgro,zg,hv,sigma_v,{gt}ij)
        Ups_der=np.ndarray((21+Nbase,3*Na,3*Na),dtype=complex)        
        E=[]
        #Generation de la famille des Ei
        for i in range(0,3):
            mat_E_tmp=np.zeros([3,3],dtype=np.complex);mat_E_tmp[i,i]=1;
            E.append(mat_E_tmp)
        
        mat_E_tmp=np.zeros([3,3],dtype=np.complex);mat_E_tmp[1,0]=1;mat_E_tmp[0,1]=1;E.append(mat_E_tmp);
        mat_E_tmp=np.zeros([3,3],dtype=np.complex);mat_E_tmp[1,0]=-1j;mat_E_tmp[0,1]=1j;E.append(mat_E_tmp);
        mat_E_tmp=np.zeros([3,3],dtype=np.complex);mat_E_tmp[2,0]=1;mat_E_tmp[0,2]=1;E.append(mat_E_tmp);
        mat_E_tmp=np.zeros([3,3],dtype=np.complex);mat_E_tmp[2,0]=-1j;mat_E_tmp[0,2]=1j;E.append(mat_E_tmp);
        mat_E_tmp=np.zeros([3,3],dtype=np.complex);mat_E_tmp[2,1]=1;mat_E_tmp[1,2]=1;E.append(mat_E_tmp)
        mat_E_tmp=np.zeros([3,3],dtype=np.complex);mat_E_tmp[2,1]=-1j;mat_E_tmp[1,2]=1j;E.append(mat_E_tmp)
        
        #k numéro de la variable par rapport à laquelle on dérive          
        for k in range(0,Nb_inc):# 0,1,...,Nb_inc
            if k in range(0,9): # 0,1,...,8            
                #derivée de T
                #Duplication de Na fois la matrice dT/dtvol,k
                T_der=I1*E[k]+0*1j
                Ups_der[k][:,:] = np.kron(np.eye(Na),T_der)
                
                #derivée de Omega_ij
                for i in range(Na-1):           
                    for j in range(i+1,Na):                                            
                        #parcours les i<j                             
                        I2gt_ij=self.get_I2(i,j)*self.get_gamma_t(i,j)
                        Om_der_ij=np.exp(1j*self.get_k_z(i,j,Na)*z_g)*I2gt_ij*E[k]
                        Ups_der[k][3*(i):3*(i+1),3*(j):3*(j+1)] = Om_der_ij
                        Ups_der[k][3*(j):3*(j+1),3*(i):3*(i+1)] = Om_der_ij.T.conj()
                                      
            elif k in range(9,18):#9,10,11,12,13,14,15,16,17
                #derivée de T
                #Duplication de Na fois la matrice dT/dtvol,k             
                T_der=a*E[k-9]
                Ups_der[k][:,:] = np.kron(np.eye(Na),T_der)
                
                #derivée par de Omega_ij
                for i in range(Na-1):
                    for j in range(i+1,Na):
                        #parcours les i<j
                        Om_der_ij=np.exp(1j*self.get_k_z(i,j,Na)*z_g)*a*E[k-9]
                        Ups_der[k][3*(i):3*(i+1),3*(j):3*(j+1)] = Om_der_ij
                        Ups_der[k][3*(j):3*(j+1),3*(i):3*(i+1)] = Om_der_ij.T.conj()

            elif k==18:                            
                #derivé par rapport à zg
                #derivée par de T
                #Duplication de Na fois la matrice dT/d(eta(k))
                T_der=np.zeros([3,3],dtype=np.complex)
                Ups_der[k][:,:] = np.kron(np.eye(Na),T_der)
                
                #derivée de Omega_ij
                for i in range(Na-1):           
                    for j in range(i+1,Na):        
                        #print i,j
                        #parcours les i<j        
                        Om_ij=Ups[3*(i):3*(i+1),3*(j):3*(j+1)]
                        Om_der_ij=1j*self.get_k_z(i,j,Na)*Om_ij
                        Ups_der[k][3*(i):3*(i+1),3*(j):3*(j+1)] = Om_der_ij
                        Ups_der[k][3*(j):3*(j+1),3*(i):3*(i+1)] = Om_der_ij.T.conj()
                
            elif k==19:    
                #Derivée par rapport à h_v
                #derivée de T
                #Duplication de Na fois la matrice dT/dtvol,k
                T_der=a*T_vol-2*sigma_v/np.cos(theta)*a*T_ground
                Ups_der[k][:,:] = np.kron(np.eye(Na),T_der)
                
                #derivée par rapport à Omega_ij
                for i in range(Na-1):           
                    for j in range(i+1,Na):                    
                        #parcours les i<j                                
                        I2gt_ij=self.get_I2(i,j)*self.get_gamma_t(i,j)
                        Om_der_ij = np.exp(1j*self.get_k_z(i,j,Na)*z_g)*(\
                                    ((-2*sigma_v/np.cos(theta))*I2gt_ij+\
                                    np.exp(1j*self.get_k_z(i,j,Na)*h_v))*T_vol-\
                                    2*sigma_v/np.cos(theta)*a*T_ground)
                                    
                        Ups_der[k][3*(i):3*(i+1),3*(j):3*(j+1)] = Om_der_ij
                        Ups_der[k][3*(j):3*(j+1),3*(i):3*(i+1)] = Om_der_ij.T.conj()                                  
                                                  
            elif k==20:
                #Dérivée par rapport à sigma_v
                der_I1=(a*(1+alpha*h_v)-1)/(sigma_v*alpha)
                der_a=-2*h_v*a/np.cos(theta)
                
                #derivée de T
                #Duplication de Na fois la matrice dT/dtvol,k
                T_der=der_I1*T_vol+der_a*T_ground
                Ups_der[k][:,:] = np.kron(np.eye(Na),T_der)
                
                #derivée de Omega_ij                
                for i in range(Na-1):           
                    for j in range(i+1,Na):                    
                        #parcours les i<j        
                        der_I2gt_ij=self.get_gamma_t(i,j)*(2/np.cos(theta)*\
                                    (a*(h_v*(1j*self.get_k_z(i,j,Na)+alpha)+1)\
                                    -np.exp(1j*self.get_k_z(i,j,Na)*h_v)))/\
                                    (1j*self.get_k_z(i,j,Na)+alpha)**2
                        Om_der_ij = np.exp(1j*self.get_k_z(i,j,Na)*z_g)*(der_I2gt_ij*T_vol+der_a*T_ground)
                        Ups_der[k][3*(i):3*(i+1),3*(j):3*(j+1)] = Om_der_ij
                        Ups_der[k][3*(j):3*(j+1),3*(i):3*(i+1)] = Om_der_ij.T.conj()
                
            elif 21<=k and k<=Nb_inc:            
                T_der = np.zeros([3,3],dtype=np.complex)
                Ups_der[k][:,:] = np.kron(np.eye(Na),T_der)    
                #derivée par rapport à Omega_ij                
                p,q=get_idx_dble_from_idx_mono(k-21,Na) #recup l'idx 2D 
                for i in range(Na-1):           
                    for j in range(i+1,Na):                    
                        Om_der_ij = self.get_I2(i,j)*T_vol*\
                                    bl.KroDelta(i,p)*bl.KroDelta(j,q)*\
                                    np.exp(1j*self.get_k_z(i,j,Na)*z_g)
                                    
                        Ups_der[k][3*(i):3*(i+1),3*(j):3*(j+1)] = Om_der_ij
                        Ups_der[k][3*(j):3*(j+1),3*(i):3*(i+1)] = Om_der_ij.T.conj()
                
        #Deduction de la M de Fisher par Slepian-Bang 
        Fisher = bl.nans([Nb_inc,Nb_inc],dtype=float)
        for a in range(0,Nb_inc):
            for b in range(0,Nb_inc):
                Fisher[a,b]=N*np.real(np.trace(bl.inv_cond(Ups,'Ups').dot(Ups_der[a][:,:].dot(bl.inv_cond(Ups,'Ups').dot(Ups_der[b][:,:])))))
        return Fisher
        
    def get_fisher_zg_known(self):
        """Retourne la matricde de Fisher à zg connu
        à partir de la matrice de Fisher         
        eta=(Tvol,Tground,hv,sigv,{gt}ij)."""        
        
        Fisher=self.get_fisher()
        Ngt = get_Nb_from_Na(self.Na)
        #Fisher -> eta=(Tvol,Tground,zg,hv,sigv,{gt}ij)
        allidx = range(21+Ngt)
        idx = allidx[:18]+allidx[18+1:]#1,2,...,21+Ngt sauf idx 18->zg
        #Suprresion de la ligne indice 18 et colonne indice 18
        Fisher_zg_known=bl.nans([20+Ngt,20+Ngt],dtype=float)
        Ftemp = Fisher[idx,:]        
        Fisher_zg_known = Ftemp[:,idx]
        return Fisher_zg_known
        
    def get_fisher_sigma_v_gt_known(self):
        """Retourne la matricde de Fisher à sigma_v et {gt}_ij connu
        eta=(Tvol,Tground,zg,hv)."""        
        
        Fisher=self.get_fisher()
        Fisher_sigma_v_gt_known=bl.nans([20,20],dtype=float)
        Fisher_sigma_v_gt_known=Fisher[0:20,0:20] #
        return Fisher_sigma_v_gt_known
        
    def get_fisher_sigma_v_known(self):
        """Retourne la matricde de Fisher à sigma_v connu         
        eta=(Tvol,Tground,zg,hv,{gt}_ij)."""
        
        Fisher=self.get_fisher()
        Ngt = get_Nb_from_Na(self.Na)
        #Fisher -> eta=(Tvol,Tground,zg,hv,sigv,{gt}ij)
        allidx = range(21+Ngt)
        idx = allidx[:20]+allidx[20+1:]#1,2,...,21+Ngt sauf idx 20->sigv
        #Suprresion de la ligne indice 20 et colonne indice 20
        Fisher_sigma_v_known=bl.nans([20+Ngt,20+Ngt],dtype=float)
        Ftemp = Fisher[idx,:]
        Fisher_sigma_v_known = Ftemp[:,idx]        
        return Fisher_sigma_v_known
        
    def get_fisher_sigma_v_Tg_Tv_known(self):
        """Retourne la matricde de Fisher à sigma_v connu 
        à partir de la matrice de Fisher à sigma_v inconnu
        eta=(zg,hv)"""
        
        Fisher_sigma_v_Tg_Tv_known=bl.nans([2,2],dtype=float)
        Fisher=self.get_fisher_sigma_v_known()        
        Fisher_sigma_v_Tg_Tv_known=Fisher[18:20,18:20]
        return Fisher_sigma_v_Tg_Tv_known
        
    def get_fisher_Tg_Tv_zg_known(self):
        """Retourne la matricde de Fisher à Tv,Tg,zg connu
        eta=(hv,sigv,{gt})
        eta=[Tvol,Tground,zg,hv,sigv,{gt}]"""
        
        Fisher=self.get_fisher()
        Ngt = get_Nb_from_Na(self.Na)
        Nbarg=2+Ngt
        Fisher_Tg_Tv_zg_known=bl.nans([Nbarg,Nbarg],dtype=float)    
        #selectionne les Nbarg avant dernieres lignes et colonnes
        Fisher_Tg_Tv_zg_known=Fisher[-Nbarg:,-Nbarg:]
        return Fisher_Tg_Tv_zg_known
        
    def get_fisher_z_c_known(self,Fisher):
        """Retourne la matricde de Fisher à z_c (haut de canopée) connue"""
        
        Fisher_z_c_known=bl.nans([20,20],dtype=float)
        #changement de var (tvol,tground,z_g,h_v,sigma_v)->(tvol,tground,z_c,h_v,sigma_v)
        J=np.eye(21);J[18,19]=1
        #mat de fisher dans les coord données nouvelles
        Fisher_2=npl.inv(J).T.dot(Fisher.dot(npl.inv(J)))
        #On supprime la 19e ligne et colonne (correspond Ã  zc)
        Idx = range(0,21)
        Idx.remove(18) #19e composante
        Fisher_z_c_known=Fisher_2[Idx,:] #suprresion 19e ligne
        Fisher_z_c_known=Fisher_z_c_known[:,Idx]#supression 19e colonne
        return Fisher_z_c_known#matrice 20x20
        
    def get_fisher_z_c_known_sigma_v_known(self,Fisher):
        """Retourne la matricde de Fisher à z_c (hauteur de canopée)
        et sigma_v (atténuation) connue à partir de la matrice 
        de Fisher à sigma_v inconnu (obtenue avec get_fisher)"""
        
        Fisher_z_c_known_sigma_v_known=bl.nans([19,19],dtype=float)
        Fisher_sigma_v_known = self.get_fisher_sigma_v_known(Fisher)
        #changement de var (tvol,tground,z_g,h_v)->(tvol,tground,z_c,h_v)
        J=np.eye(20);J[18,19]=1
        #mat de fisher dans les coord données nouvelles
        Fisher_2_sigma_v_known=npl.inv(J).T.dot(Fisher_sigma_v_known.dot(npl.inv(J)))
        #On supprime la 19e ligne et 19e colonne (correspond a zc)
        Idx = range(0,20)
        Idx.remove(18) #19e composante
        Fisher_z_c_known_sigma_v_known= \
                  Fisher_2_sigma_v_known[Idx,:] #supression 19e ligne
        Fisher_z_c_known_sigma_v_known= \
                  Fisher_z_c_known_sigma_v_known[:,Idx]#supression 19e colonne        
        return Fisher_z_c_known_sigma_v_known
        
def get_idx_dble_from_idx_mono(k,Na):    
    """ Renvoie l'index 2D de la keme réponse interferometrique
    
    ex: En Dual-Baseline (Na=3)
    
    * 0 -> (0,1) 
    * 1 -> (0,2) 
    * 2 -> (1,2)         
        
    **Entrées** 
    
    * **k** : index 1D 
    * **Na** : nombre d'antennes"""    
    
    compt=0
    Nbase=get_Nb_from_Na(Na)
    if k<0 or k>Nbase-1:
        print 'Erreur: index impossible'
    for i in range(Na-1):           
        for j in range(i+1,Na):
            if compt==k:            
                return (i,j)
            else:
                compt=compt+1
    
def get_Nb_from_Na(Na):
    """Renvoie le nombre de baseline possible à partir de Na antennes"""
    return int(np.floor(Na*(Na-1)/2))
    
def get_Na_from_Nb(Nb):
    """Renvoie le nombre d'antenne à partir du nombre de baselines"""
    return int(np.floor((1.+np.sqrt(1.+8*Nb))/2))
    
def rvog_reduction(param,A,E):
    """Modifie le vecteur de paramètre RVoG et renvoie un vecteur de paramètres
    correspondant au modèle RVoG réduit 
        
    **Entrées**
    
    * **A**: paramètre (invariant) de contraste
    * **E**: paramètre (invariant) 
    * **param** : paramètre type param_rvog
    
    
    **Sortie**
           
    * **param_out** : paramètre type param_rvog
    
    N.B1: on fixe lmda2 = (lmda1+lmda2) /2 
    car (cf Artciel P.Refregier )"Invariant Contrast Parameters of PolInSARHomogenous RVoG Model"
    lmbda2 ne modifie quasiment pas la BCR   
    N.B2: dans param seuls T_vol,T_ground et z_g sont modifiés 
    """  
    
    lmbda=np.zeros(3)    
    lmbda[0]=E/3*(A+1)
    lmbda[1]=E/3
    lmbda[2]=E/3*(1-A)
    T_vol_red=np.eye(3)    
    T_ground_red=np.diag(lmbda)
    z_g_red=0
    
    param_out = cp.copy(param)
    param_out.T_vol=T_vol_red
    param_out.T_ground=T_ground_red
    param_out.z_g=z_g_red
    return param_out    
    
def gamma_v_kz(param,k_z):
    """Renvoie gamma_v en fonction de k_z"""
    
    a=np.exp((-2*param.sigma_v*param.h_v)/(np.cos(param.theta)))
    alpha=2*param.sigma_v/np.cos(param.theta)    
    I1=(1-a)/alpha
    I2 = (np.exp(1j*k_z*param.h_v)-a)/\
                (1j*k_z+alpha)
    gamma_v=I2/I1
    return gamma_v
    
def rvog_bcr_h_v(param):    
    """Calcul des bcr h_v pour differentes connaissances à priori
    BCR[h_v|sigma_v] BCR[h_v|z_c] et BCR[h_v|sigma_v,z_c]
    en fonction de h_v
    
    **Entrées**    
        * **param** : paramètre type param_rvog
    
    **Sorties**     
        * **bcr_h_v_sigma_v_known**
        * **bcr_h_v_z_c_known**
        * **bcr_h_v_z_c_known_sigma_v_known**"""
      
    Fisher=param.get_fisher()
    Fisher_sigma_v_known=param.get_fisher_sigma_v_known(Fisher)
    Fisher_z_c_known_sigma_v_known=param.get_fisher_z_c_known_sigma_v_known(Fisher)
    Fisher_z_c_known=param.get_fisher_z_c_known(Fisher)
    
    mat_BCR_sigma_v_known=bl.inv_cond(Fisher_sigma_v_known)
    mat_BCR_z_c_known=bl.inv_cond(Fisher_z_c_known)
    mat_BCR_z_c_known_sigma_v_known=bl.inv_cond(Fisher_z_c_known_sigma_v_known)
    
    bcr_h_v_sigma_v_known=mat_BCR_sigma_v_known[19,19]        
    bcr_h_v_z_c_known=mat_BCR_z_c_known[18,18]#19e composante                
    bcr_h_v_z_c_known_sigma_v_known=mat_BCR_z_c_known_sigma_v_known[18,18]        
    
    return bcr_h_v_sigma_v_known,bcr_h_v_z_c_known,bcr_h_v_z_c_known_sigma_v_known
    
def rvog_bcr_sigma_v(param):    
    """Calcul de la bcr sigma_v pour z_c connue (hauteur de canopée)
    BCR[sigma_v|z_c]  en fonction de h_v
    
    **Entrée**     
        * **param** : paramètre type param_rvog
    
    **sortie**    
        * **bcr_sigma_v_z_c_known**
    
    *N.B*: on peut estimer sigma_v si on connait z_c"""
    
    Fisher=param.get_fisher()
    Fisher_z_c_known=param.get_fisher_z_c_known(Fisher)
    
    mat_BCR_z_c_known=bl.inv_cond(Fisher_z_c_known)   
    bcr_sigma_v_z_c_known=mat_BCR_z_c_known[19,19]#20e composante
   
    return bcr_sigma_v_z_c_known
    
def plot_rvog_bcr_h_v(param,A,E,vec_h_v):
    """Trace les 3 BCR de h_v (BCR[h_v|sigma_v] BCR[h_v|z_c] et BCR[h_v|sigma_v,z_c])
    pour des valeurs de A et E fixées. 
    
    **Entrées**        
        * **param** : paramètre de type param_rvog        
        * **A**,**E** : paramètres invariants
        * **vec_h_v** : vecteur contenant la valeur que prend h_v"""
        
    #obtention du model rvog réduit
    param_red = rvog_reduction(param,A,E)
    #obtention des listes de bcr 
    bcr_h_v_sigma_v_known=[]
    bcr_h_v_z_c_known=[]
    bcr_h_v_z_c_known_sigma_v_known=[]
    for i,h_v in enumerate(vec_h_v):
        param_red.h_v=h_v
        bcr_tmp1,bcr_tmp2,bcr_tmp3=rvog_bcr_h_v(param_red)
        bcr_h_v_sigma_v_known.append(bcr_tmp1)        
        bcr_h_v_z_c_known.append(bcr_tmp2)#19e composante                
        bcr_h_v_z_c_known_sigma_v_known.append(bcr_tmp3)
   
    plt.figure(num=1,figsize=(16,14),dpi=80)

    plt.semilogy(vec_h_v,np.real(bcr_h_v_sigma_v_known),'ob',label = 'BCR[hv|sigv]')
    plt.semilogy(vec_h_v,np.real(bcr_h_v_z_c_known),'or', label = 'BCR[hv|zc]')
    plt.semilogy(vec_h_v,np.real(bcr_h_v_z_c_known_sigma_v_known),'og', label = 'BCR[hv|zc,sigv]')                

    plt.xlabel('hv',fontsize=20);plt.ylabel('BCR',fontsize=20)
    plt.grid('True',which='minor',ls='--',alpha=0.3)
    plt.grid('True',which='major',ls='--',alpha=0.5)
    plt.legend(loc='lower right')
    plt.title('BCR[hv] \n A= '+str(A)+' E= '+str(E) ,fontsize=20)
    
    #Test pour légender les bcr
    idx_sig=np.int(np.floor(param.N/2))
    idx_zc=idx_sig-20
    idx_zcsig=idx_sig+20
    x_sig= vec_h_v[idx_sig];x_zc= vec_h_v[idx_zc];x_zcsig= vec_h_v[idx_zcsig]
    y_sig= np.real(bcr_h_v_sigma_v_known[idx_sig])
    y_zc= np.real(bcr_h_v_z_c_known[idx_zc])
    y_zcsig = np.real(bcr_h_v_z_c_known_sigma_v_known[idx_zcsig])
    
    plt.annotate('BCR[hv|sigv]', 
                 xy=(x_sig,y_sig), xycoords='data',
                 xytext = (1,-80), textcoords='offset points',fontsize=16,
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate('BCR[hv|zc]', 
                 xy=(x_zc,y_zc), xycoords='data',
                 xytext = (15,-120), textcoords='offset points',fontsize=16,
                 arrowprops=dict(arrowstyle="->"))
                 #pr faire di jouli arc connectionstyle="arc3,rad=0.2")
    plt.annotate('BCR[hv|zc,sigv]', 
                 xy=(x_zcsig,y_zcsig), xycoords='data',
                 xytext = (20,-160), textcoords='offset points',fontsize=16,
                 arrowprops=dict(arrowstyle="->"))
    
def plot_rvog_bcr_sigma_v(param,A,E,vec_h_v):                 
    """Trace la BCR de sigma_v 
    pour des valeurs de A et E fixées. 
    
    **Entrées**    
        * **vec_h_v** : vecteur contenant la valeur que prend h_v
        * **param** : paramètre de type param_rvog"""
    
    #obtention du model rvog réduit
    param_red = rvog_reduction(param,A,E)    
    #obtention de la bcr 
    bcr_sigma_v=[]
    for i,h_v in enumerate(vec_h_v):
        param_red.h_v=h_v
        bcr_sigma_v.append(rvog_bcr_sigma_v(param_red))
    plt.figure(num=2,figsize=(16,14),dpi=80)
    plt.semilogy(vec_h_v,np.real(bcr_sigma_v),'ob',label = 'BCR[sigv|zc]')
    plt.xlabel('hv',fontsize=20);plt.ylabel('BCR',fontsize=20)
    plt.grid('True',which='minor',ls='--',alpha=0.3)
    plt.grid('True',which='major',ls='--',alpha=0.5)
    plt.legend(loc='upper left')
    plt.title('BCR[sigv|zc] \n A= '+str(A)+' E= '+str(E) ,fontsize=20)    

def plot_rvog_bcr_sigma_v_hsig(param,A,E,vec_h_v,vec_sigma_v):                 
    """Trace la bcr sigma_v
    pour des valeurs de A et E fixées en fonction de sigma_v*h_v.
    ceci pour plusieurs valeurs de sigma_v
    
    **Entrées** : 
        * **vec_h_v** : vecteur contenant la valeur que prend h_v
        * **param** : paramètre de type param_rvog"""
    
    #obtention du model rvog réduit
    param_red = rvog_reduction(param,A,E)    
    #obtention de la bcr 
    bcr_sigma_v= [[]for i in range(vec_sigma_v.size)]
    
    
    plt.figure(num=2,figsize=(16,14),dpi=80)    
    plt.xlabel('hv',fontsize=20);plt.ylabel('BCR',fontsize=20)
    plt.grid('True',which='minor',ls='--',alpha=0.3)
    plt.grid('True',which='major',ls='--',alpha=0.5)
    plt.legend(loc='upper left')
    plt.title('BCR[sigv|zc] \n A= '+str(A)+' E= '+str(E) +
              '\n sigv=' + str(vec_sigma_v),fontsize=20)    
    
    for k,sigma_v in enumerate(vec_sigma_v):
        param_red.sigma_v=vec_sigma_v[k]
        for i,h_v in enumerate(vec_h_v):
            param_red.h_v=h_v
            bcr_sigma_v[k].append(rvog_bcr_sigma_v(param_red))   
    
        plt.semilogy(vec_sigma_v[k]*vec_h_v,vec_h_v*np.real(bcr_sigma_v[k]),'ob',label = 'BCR[sigv|zc]')
        
        plt.annotate('sig='+str(vec_sigma_v[k]), 
                 xy=(vec_h_v[-1]*vec_sigma_v[k],np.real(bcr_sigma_v[k])[-1]), xycoords='data',
                 xytext = (1,1), textcoords='offset points',fontsize=16,
                 arrowprops=dict(arrowstyle="->"))
        
def bcr_hv_mb(param):
    """Renvoie la bcr de hv dans un cas multibaseline
    eta=(Tvol,Tground,zg,hv,sigv,{gt}ij) 
    avec {gt}ij l'ens des décohérences temporelle i<j"""
    
    Fish_MB= param.get_fisher()
    Inv_MB = bl.inv_cond(Fish_MB,name='Fish_MB')
    BCR_hv_MB = np.real(Inv_MB[19,19])
    return BCR_hv_MB
    
def bcr_hv_mb_zg_connu(param):
    """Renvoie la bcr de hv dans un cas multibaseline à zg connu
    eta=(Tvol,Tground,hv,sigv,{gt}ij) 
    avec {gt}ij l'ens des décohérence temporelle i<j"""        
    
    F = param.get_fisher_zg_known()   
    invF = npl.inv(F)
    bcrhv = np.real(invF[18,18])#19e composante
    return bcrhv
    
def bcr_hv_mb_Tv_Tg_zg_connu(param):
    """Renvoie la bcr de hv a Tv,Tg,zg connu 
    eta=(hv,sigv,{gt}ij)."""
    
    F = param.get_fisher_Tg_Tv_zg_known()
    invF = bl.inv_cond(F,name='F_Tv_Tg_zg_connu')
    bcrhv = np.real(invF[0,0])
    return bcrhv
    
def bcr_hv_mb_sigv_connu(param):
    """Renvoie la bcr de hv dans le cas sigma_v connu
    dans un cas multibaseline.
    
    vect param : eta=(tvol,tgro,zg,hv)"""
    
    Fish_MB= param.get_fisher_sigma_v_known2()
    Inv_MB = bl.inv_cond(Fish_MB,name='Fish_MB_sigv_connu')
    BCR_hv_MB = np.real(Inv_MB[19,19])
    return BCR_hv_MB

def bcr_hv_mb_sigv_inconnu(param):
    """Renvoie la bcr de hv dans le cas sigma_v inconnu
    dans un cas multibaseline.
    
    vect param : eta=(tvol,tgro,zg,hv,sigv)"""
    
    Fish_MB= param.get_fisher2()
    Inv_MB = bl.inv_cond(Fish_MB,name='Fish_MB')
    BCR_hv_MB = np.real(Inv_MB[19,19])
    return BCR_hv_MB
    
def bcr_hv_mb_Tv_Tg_sigv_gt_known(param):
    """Renvoie la bcr de hv dans le cas Tv,Tg,sigma_v connu et gamma_t connu 
    dans un cas multibaseline.
    
    vect param : eta=(zg,hv)"""   
    
    Fish = param.get_fisher_sigma_v_Tg_Tv_known2()
    Inv = bl.inv_cond(Fish,name='Fish_Tv_Tg_sigv_known')
    BCR_hv = np.real(Inv[1,1])
    return BCR_hv
    
def Tv_Tg_from_A_E_X(A,E,X):
    """Genere T_vol et T_ground du modèle RVoG à partir des paramètres invariants
    A, E et X
    
    * A=(l1-l3)/(l1+l3)
    * E=l1l+l2+l3
    * X=(l2-l3)/(l1+l3) (l=lambda)"""
    
    lmbda=np.zeros((3),dtype=float)
    D = 3+A*(2*X-1)
    lmbda[0]=1/D*E*(A+1)
    lmbda[1]=1/D*E*(1+A*(2*X-1))
    lmbda[2]=1/D*E*(1-A)
    Tground = np.diag(lmbda)
    Tvol=np.eye(3)
    return Tvol,Tground
    
def get_vecTm_from_Tm(Tm):
    """Renvoie sur forme de vecteurs les coeff de la matrice Tm    
    selon l'odre défini dans la These d'A.Arnaubec"""
    
    vec_Tm = np.zeros(9,dtype=float)
    vec_Tm[0] = np.real(Tm[0,0])
    vec_Tm[1] = np.real(Tm[1,1])
    vec_Tm[2] = np.real(Tm[2,2])
    vec_Tm[3] = np.real(Tm[0,1])
    vec_Tm[4] = np.imag(Tm[0,1])
    vec_Tm[5] = np.real(Tm[0,2])
    vec_Tm[6] = np.imag(Tm[0,2])
    vec_Tm[7] = np.real(Tm[1,2])
    vec_Tm[8] = np.imag(Tm[1,2])
    return vec_Tm
    
def get_Tm_from_vecTm(vec_Tm):
    """Renvoie sur forme de vecteurs les coeff de la matrice Tm    
    selon l'odre défini dans la These d'A.Arnaubec"""
    
    Tm = np.zeros((3,3),dtype=complex)
    Tm[0,0] = np.real(vec_Tm[0])     
    Tm[1,1] = np.real(vec_Tm[1])     
    Tm[2,2] = np.real(vec_Tm[2])     
    Tm[0,1] = vec_Tm[3]+1j*vec_Tm[4]
    Tm[1,0] = vec_Tm[3]-1j*vec_Tm[4]
    Tm[0,2] = vec_Tm[5]+1j*vec_Tm[6]
    Tm[2,0] = vec_Tm[5]-1j*vec_Tm[6]
    Tm[1,2] = vec_Tm[7]+1j*vec_Tm[8]
    Tm[2,1] = vec_Tm[7]-1j*vec_Tm[8]
    return Tm
    
def bcr_mb_A_influence(param,vec_A,E,X,vec_hv,scenario):
    """Calcul de la bcr de hv pour differents A (tout fixé par ailleurs).
    
    Entrées

    * **param** : parametre RVoG 
    * **vec_A** : vecteur des valeurs du paramètre invariant A
    * **vec_hv** : vecteur de variation de h_v
    * **scenario** : scenarios possibles 
    
        * *sigv_connu* : sigma_v connu -> eta =(tvol,tgro,hv,zg,sigv)
        * *sigv_inconnu* : sigma_v inconnu -> eta =(tvol,tgro,hv,zg)
        
    """
        
    BCR_hv_MB=np.zeros((np.shape(vec_hv)[0],len(vec_A)))    
    #matr_Cond=np.zeros((np.shape(vec_hv)[0],len(list_kz)))
    
    for (idx_A,A) in enumerate(vec_A):
        param.T_vol,param.T_ground=Tv_Tg_from_A_E_X(A,E,X)
        for (idx_hv,hv) in enumerate(vec_hv):                    
            param.h_v=hv
            if scenario == 'sigv_connu':
                BCR_hv_MB[idx_hv,idx_A]=bcr_hv_mb(param) 
            elif scenario == 'sigv_inconnu':
                BCR_hv_MB[idx_hv,idx_A]=bcr_hv_mb_sigv_inconnu(param) 
            else:
                print "Pass de scénario reconnu"
    return BCR_hv_MB            
    
def bcr_mb_kz_influence(param,A,E,X,vec_hv,list_kz,scenario):    
    """Calcul de la bcr de hv pour differents kz (tout fixé par ailleurs).
    
    Entree
    
    * **param** : parametre RVoG 
    * **vec_A** : vecteur des valeur du paramètre invariant A
    * **vec_hv** : vecteur de variation de h_v
    * **list_kz** : liste dont chaque element contient une liste de kz 
    * **scenario** : scenarios possibles 
    
        * *sigv_connu* : sigma_v connu -> eta =(tvol,tgro,hv,zg,sigv)
        * *sigv_inconnu* : sigma_v inconnu -> eta =(tvol,tgro,hv,zg)
        
    """
    
    BCR_hv_MB=np.zeros((np.shape(vec_hv)[0],len(list_kz)))    
    #matr_Cond=np.zeros((np.shape(vec_hv)[0],len(list_kz)))
    param.T_vol,param.T_ground=Tv_Tg_from_A_E_X(A,E,X)
    
    for idx_kz in range(len(list_kz)):
        param.k_z=list_kz[idx_kz]
        param.Na=len(param.k_z)+1
        
        for (idx_hv,hv) in enumerate(vec_hv):                    
            param.h_v=hv
            if scenario == 'sigv_connu':
                BCR_hv_MB[idx_hv,idx_kz]=bcr_hv_mb(param) 
            elif scenario == 'sigv_inconnu':
                BCR_hv_MB[idx_hv,idx_kz]=bcr_hv_mb_sigv_inconnu(param) 
            else:
                print "Pass de scénario reconnu"
            #matr_Cond[idx_hv,idx_kz]=bcr_hv_mb(param)          
            #BCR_hv_MB[idx_hv,idx_kz]=bcr_hv_mb_Tv_Tg_sigv_known(param)            
            #matr_Cond[idx_hv,idx_kz]=bcr_hv_mb_Tv_Tg_sigv_known(param)          
            
    return BCR_hv_MB      

def bcr_mb_N_influence(param,A,E,X,vec_N,scenario):       
    
    """Calul la BCR en mb de hv pour differentes taillles d'échantillon (vec_N)

    Entrées
            
    * **param** object paramètre RVoG mb\n
    * **A**,**E**,**X** paramètres invariants de la BCR\n
    * **vec_N** vec contenant les tailles d\'echantillons testées    
    * **scenario** : scenarios possibles 
    
        * *sigv_connu* : sigma_v connu -> eta =(tvol,tgro,hv,zg,sigv)
        * *sigv_inconnu* : sigma_v inconnu -> eta =(tvol,tgro,hv,zg)
                    
    """

    BCR_hv_MB=np.zeros((np.shape(vec_N)[0],1))        
    param.T_vol,param.T_ground=Tv_Tg_from_A_E_X(A,E,X)
           
    for (idx_N,N) in enumerate(vec_N):                    
        param.N=N
        if scenario == 'sigv_connu':
            BCR_hv_MB[idx_N]=bcr_hv_mb(param) 
        elif scenario == 'sigv_inconnu':
            BCR_hv_MB[idx_N]=bcr_hv_mb_sigv_inconnu(param) 
        else:
            print "Pass de scénario reconnu"
            
    return BCR_hv_MB             
    
def plot_and_save_bcr_mb_N_influence(param,A,E,X,vec_N,scenario,
                                    save=True,foldername=''):
    """ plot la sqrt(bcr hv) (en MB) pour differents tailles d'echant (vec_N)
    
    Entrées :
    
    * **param** : paramètre RVoG mb
    * **A**,**E**,**X** : paramètres invariants de la BCR 
    * **vec_N** : vecteur de tailles d'échantillon              
    * **scenario** : scenarios possibles 
    
        * *sigv_connu* : sigma_v connu -> eta =(tvol,tgro,hv,zg,sigv)
        * *sigv_inconnu* : sigma_v inconnu -> eta =(tvol,tgro,hv,zg)
    
    * **save**,**foldername** : sauvegarde des plot dans *filename* si *save* =True
    """    
    
    BCR_hv_MB=bcr_mb_N_influence(param,A,E,X,vec_N,scenario)    
                    
    bl.pplot(vec_N,BCR_hv_MB,names='')
    if save:
        str_save_info='_'.join([bl.strval(A,'A'),bl.strval(E,'E')])
        filename=foldername+'\\BCR_hv_'+str_save_info+'.png'
        bl.save_fig(filename=filename)
  
 
def plot_bcr_mb_kz_influence(param,A,E,X,vec_hv,list_kz,scenario):
    """ plot la sqrt(bcr hv) (en MB) pour differents Nb/tailles de baselines
    stockées dans list_kz en fcontion de hv pour des param A, E et X fixés.
    
    Entrées
    
    * **param** : object paramètre RVoG mb
    * **A**,**E**,**X** : paramètres invariants de la BCR 
    * **vec_hv** : variation de hv (axe X du plot)
    * **list_kz** : liste contenant la configuration sur chaque element:
            
                        chaque élement doit contenir les kz_1j\n
                        ex : pour plot la bcr dans le cas SB\n
                        puis dual baseline: list_kz=[[kz12],[kz12,kz13]]
                        
    * **scenario** : scenarios possibles 
    
        * *sigv_connu* : sigma_v connu -> eta =(tvol,tgro,hv,zg,sigv)
        * *sigv_inconnu* : sigma_v inconnu -> eta =(tvol,tgro,hv,zg)
    """    
    
    BCR_hv_MB=bcr_mb_kz_influence(param,A,E,X,vec_hv,list_kz,scenario)    
    #liste de nom : chq courbe identifié par sa liste de kz
    str_name=['kz_1n='+'_'.join(['{:2.2f}'.format(list_kz[i][j]) 
                for j in range(len(list_kz[i]))]) for i in range(len(list_kz))]
    bl.pplot(vec_hv,np.sqrt(BCR_hv_MB),names=str_name)
       
def plot_and_save_bcr_mb_A_E_influence(param,vec_A,vec_E,X,
                                                   vec_hv,
                                                   save=True,scenario='',
                                                   foldername=''):                                                           
    """
    Trace sqrt(bcr hv) (cas MB).
    
    Plot la sqrt(bcr hv) (en MB) pour differents A et E sur une mm graphe.
    Pour une val de E, les bcr f(hv,A,E) sont la mm couleur.
    
    
    Entrées
            
    * **param** object paramètre RVoG mb
    * **vec_A**,**vec_E**,**X**: paramètres invariants de la BCR\n
    * **vec_hv** : variation de hv (axe X du plot)
    * **save** : boolen = True pour sauver (False sinon)
    * **foldername** : chemin du dossier où sauver l'image\n
    * **scenario** : scenarios possibles 
    
        * *sigv_connu* : sigma_v connu -> eta =(tvol,tgro,hv,zg,sigv)
        * *sigv_inconnu* : sigma_v inconnu -> eta =(tvol,tgro,hv,zg)
    """   
    
    list_BCR_hv_MB=[]
    list_name=[]
    for (idx_E,E) in enumerate(vec_E):
        list_BCR_hv_MB.append(np.sqrt(bcr_mb_A_influence(param,vec_A,E,X,vec_hv,scenario)))            
        list_name.append(['\n'.join((bl.strval(vec_A[i],'A'),bl.strval(E,'E'))) for i in range(len(vec_A))])
       
    str_save_info='A='+'_'.join([bl.strval(vec_A[i]) for i in range(len(vec_A))])\
                 +'_E='+'_'.join([bl.strval(int(vec_E[i])) for i in range(len(vec_E))])\
                 +'_kz='+'_'.join([bl.strval(param.k_z[i])for i in range(len(param.k_z))])
                 
    bl.pplot_monochrome_blocs(vec_hv,list_BCR_hv_MB,
                               bloc_list_names=list_name)
    
    if save:
        filename=foldername+'\\BCR_hv_'+str_save_info+'.png'
        bl.save_fig(filename=filename)






def plot_and_save_bcr_mb_kz_influence_A_E_variable(param,vec_A,vec_E,X,
                                                   vec_hv,list_kz,
                                                   save=True,scenario='',
                                                   foldername=''):                                                           
    """
    Trace sqrt(bcr hv) (cas MB).
    
    Plot la sqrt(bcr hv) (en MB) pour differents Nb/tailles de baselines
    stockées dans list_kz en fcontion de hv pour des param A, E et X fixés 
    stockées dans vec_A et vec_E. pour chaque couple (A,E) on obtient 
    une figure BRhv = f(hv) pour la liste des kz contenues dans list_kz.
    
    Entrées
            
    * **param** object paramètre RVoG mb\n
    * **A,E,X** paramètres invariants de la BCR\n
    * **vec_hv** variation de hv (axe X du plot)\n
    * **list_kz** liste contenant la configuration sur chaque element\n
              chaque élement doit contenir les kz_1j\n
              ex : pour plot la bcr dans le cas SB\n
              puis dual baseline: list_kz=[[kz12],[kz12,kz13]]\n
    * **save** boolen = True pour sauver (False sinon)\n
    * **foldername** chemin du dossier où sauver l'image\n
    * **scenario** : scenarios possibles 
    
        * *sigv_connu* : sigma_v connu -> eta =(tvol,tgro,hv,zg,sigv)
        * *sigv_inconnu* : sigma_v inconnu -> eta =(tvol,tgro,hv,zg)                    
    """
    
    param.k_z=list_kz[-1]#Seulemnt pour ecriture de la config 
    bl.save_txt_config(param,foldername,name='config',showTvTg=False)
    for (idx_A,A) in enumerate(vec_A):
        for (idx_E,E) in enumerate(vec_E):
            
            BCR_hv_MB=bcr_mb_kz_influence(param,A,E,X,vec_hv,list_kz,scenario)                            
            kz_name=['kz_1n='+'_'.join(['{:2.2f}'.format(list_kz[i][j]) 
                for j in range(len(list_kz[i]))]) for i in range(len(list_kz))]
            #kz_name[-1] contient 'k12,...kz1n
            str_save_info='_'.join([bl.strval(A,'A'),bl.strval(E,'E'),kz_name[-1]])
            
            bl.pplot(vec_hv,np.sqrt(BCR_hv_MB),names=kz_name)
            if save:
                bl.save_fig(filename=foldername+'\\'+str_save_info+'.png')
    
