# -*- coding: utf-8 -*-
""" Classe TomoSARDataSet_synth et routines associées"""
from __future__ import division
import plot_tomo as pt
import sys
import numpy as np
import numpy.linalg as npl
import matplotlib.mlab as mat
import matplotlib.pyplot as plt
import stat_SAR as st
import RVoG_MB as mb
import basic_lib as bl
import os 
import pdb
plt.ion()

class TomoSARDataSet_synth:
    """ Classe TomoSARDataSet_synth permettant de stocker/analyser
        des données mutlibaseline-polarimetrique."""
    def __init__(self,param):
        """Initialisation de la classe à l'aide d'un object type param_rvog."""
        self.k_z = param.k_z
        self.theta = param.theta
        self.T_vol = param.T_vol
        self.T_ground = param.T_ground
        self.h_v = param.h_v
        self.z_g = param.z_g
        self.extinction = param.sigma_v
        #Na nombre d'antennes
        self.Na = param.Na
        #Données
        self.imgMhh = [None]*self.Na
        self.imgMhv = [None]*self.Na
        self.imgMvh = [None]*self.Na
        self.imgMvv = [None]*self.Na
        self.Ha = [None]*self.Na
        
    def get_data_teb_rect(self, nb_echant):                        
        """Récupère un tableau de dimension *(Nax3)xnb_echant* contenant 
        les valeurs des pixels associés aux données dans la base MPMB.
        
        Les données sont exprimées dans la base MPMB:\n
        y = (hh1,hh2,..,hhNa,hv1,hv2,..,hvNa,vv1,vv2,..,vvNa)\n
        **Entrée** : - nb_echant : taille de l'echantillon \n
        **Sortie** : - data : matrice (3*Na,nb_echant) contenant les données 
        """
        
        Na = self.Na
        taille = nb_echant
        data = np.zeros((3*Na,taille), 'complex64')
        for j in range(Na):
            data[j,:] = self.imgMhh[j][0:taille]            
            data[j+Na,:] = self.imgMhv[j][0:taille]
            data[j+2*Na,:] = self.imgMvv[j][0:taille]
        return data
        
    def get_data_rect(self, nb_echant):
        """Récupère un tableau de dimension *(3xNa)xnb_echant* contenant 
        les valeurs des pixels associes aux données en base lexicographique.
            
            Les données sont exprimées dans la base lexicographique:
            y = (hh1,hv1,vv1,hh2,hv2,vv2,...,hhNa,hvNa,vvNa)
            **Entrée** : - nb_echant : taille de l'echantillon \n
            **Sortie** : - data : matrice (3*Na,nb_echant) contenant les données 
        """
        
        Na = self.Na
        taille = nb_echant
        data = np.zeros((3*Na,taille), 'complex64')
        if (taille > self.N): 
            print "Taille d'echantillon trop grande!!"        
        #cas monostatique : hv = vh 
        for j in range(Na):
            data[3*j,:] = self.imgMhh[j][0:taille]            
            data[3*j+1,:] = self.imgMhv[j][0:taille]
            data[3*j+2,:] = self.imgMvv[j][0:taille]
        return data
      
    def get_ha_single_rect(self,ant1,ant2):
        """Renvoie la hauteur d'ambigüité entre les antennes *ant1* et *ant2*.
        
        **Entrées**: -*anti* : indice de l'antenne *i* (doit appartenir à *[0,Na-1]*)\n
        **Sortie**: -*ha_single* : hauteur d'ambiguité
        """
        
        H=self.Ha
        sign=0        
        if (ant1>ant2):
            ant1,ant2 = (ant2,ant1)
            sign = -1 
        elif (ant1<ant2):
            sign = 1
        else:
            print 'Dans get_ha_single_rect'
            print "Attention! : Même valeur! ant1=",ant1,"ant2=",ant2
        ha_single= sign*1/((-1/H[ant1]+1/H[ant2]))
        return ha_single
        
    def get_k_z(self,ant1,ant2):
        """Renvoie la sensibilité interferometrique
        k_z entre les anntennes *ant1* et *ant2*.
        
        **Entrées**: -*anti*: indice de l'antenne *i* (doit appartenir à [0,Na-1])                
        **Sortie**: -*k_z_ant1_ant2*: sensibilité interfométrique entre *ant1* et *ant2*
        """
        
        k_z = self.k_z        
        k_z_ant1_ant2 = -k_z[ant1]+k_z[ant2]
        return k_z_ant1_ant2 
    
    def get_W_k_rect(self,param,nb_echant):
        """Renvoie la matrice de covariance en coordonnées MPMB.
        
        **Entrées** 
            * *param* : classe de paramètre rvog
            * *nb_echant* : taille d'echantillon
            
        **Sortie**: 
            * *covar* : matrice de covariance
        """
        
        Ups = param.get_upsilon_gt()
        k = st.generate_PolInSAR(Ups,nb_echant)        
        covar_lexi = k.dot(k.T.conj())/nb_echant
        covar_mpmb = UPS_to_MPMB(covar_lexi)
        return covar_mpmb
    
    def get_W_k_norm_rect(self,param,nb_echant,type_norm='ps+tebald'):
        """Renvoie la matrice de covariance normalisée en coordonnées MPMB.
        
        **Entrées**:
            * *param*: classe de paramètre rvog
            * *nb_echant*: taille d'echantillon            
            * *type_norm*: type de normalisation 
            
                * *mat+ps*: Application d'une normalisation+egalisation des rep polarmietrique
                * *mat+ps+tebald*: precedent + normalisation selon chque recepteur et polar
                * *ps+tebald*: egalisation des reps polar+tebald
                
        **Sortie**: 
            * *covar* : matrice de covariance
        """
        
        Na = self.Na
        covar_mpmb = self.get_W_k_rect(param,nb_echant,Na)
        
        if type_norm =='mat+ps':
            covar_norm = normalize_MPMB_mat_PS(covar_mpmb,Na)
        elif type_norm =='ps+tebald':
            covar_norm = normalize_MPMB_PS_Tebald(covar_mpmb,Na)
        elif type_norm == 'mat+ps+tebald':
            covar_norm = normalize_MPMB_mat_PS_Tebald(covar_mpmb,Na)
        else:
            print 'Attention Type de normalisation inconnu ! '
        return covar_norm
        
    def get_covar_rect(self,param,nb_echant):
        #Na = self.Na
        covar_mpmb = self.get_W_k_rect(param,nb_echant)
        covar_lexico = MPMB_to_UPS(covar_mpmb)
        """
        data = self.get_data_rect(nb_echant,Na)
        covar = data.dot(data.T.conj())/nb_echant
        """
        return covar_lexico      
        
def normalize_MPMB_mat_PS(W,Na):         
    """Normalisation de la matrice de covariance pour imposer 
    la stationnarité polarimétrique (PS)
    (hh1=hh2=..=hhNa; hv1=hv2...=hvNa et vv1=vv2=...=vvNa)
    
    Version inspirée de la méthode de Pascale:\n
    #. Passage dans la base lexicographie 
    #. Normalisation (operation matricielle)
    #. Passage base MPMB (operation matricielle)
      
    **Entrées**: 
        * *W* : matrice de covariace (base MPMB)
        * *Na* : nombre d'antennes

    **Sortie**:
        * *W_cal* : matrice de covariance normalisée (base MPMB)
    """    
    
    Cal = np.zeros((3*Na,3*Na),'float')
    Ups_cal = np.zeros((3*Na,3*Na),'complex64')
    Ups_cal2 = np.zeros((3*Na,3*Na),'complex64')
    cal_hh = np.zeros((Na-1,1),'float')
    cal_hv = np.zeros((Na-1,1),'float')
    cal_vv = np.zeros((Na-1,1),'float')
    T = np.zeros((Na,3,3),'complex64')
    Ups = MPMB_to_UPS(W,Na)    
    for i in range(Na-1):
        cal_hh[i]=np.sqrt(np.real(Ups[0,0]/Ups[3*(i+1),3*(i+1)]))
        cal_hv[i]=np.sqrt(np.real(Ups[1,1]/Ups[3*(i+1)+1,3*(i+1)+1]))
        cal_vv[i]=np.sqrt(np.real(Ups[2,2]/Ups[3*(i+1)+2,3*(i+1)+2]))
    vec_diag_cal = np.vstack((np.vstack((cal_hh,cal_hv)),cal_vv))
        
    Cal[0:3,0:3] = np.eye(3) 
    Cal[3:,3:] = np.diagflat(vec_diag_cal)
    Ups_cal = Cal.dot(Ups.dot(Cal))
    
    Ups_cal2 = Ups_cal.copy()
    for i in range(Na):
        T[i][:,:] = Ups_cal[i*3:i*3+3,i*3:i*3+3].copy()
        
    mean_T = T.mean(0)
    for i in range(Na):
        Ups_cal2[i*3:i*3+3,i*3:i*3+3] = mean_T
        
    W_cal = UPS_to_MPMB(Ups_cal2)
    return W_cal
    
def normalize_MPMB_mat_PS_Tebald(W,Na):
    """Applique la normalisation  \'mat+ps\' puis celle de Tebaldini.
    
    Normalisation tebaldini: normalisation selon chaquee canal et recepteur 
    Gamma = (E-1/2 Ups E-1/2 avec E=diag(Ups))
    
   **Entrées**: 
        * *W* : matrice de covariace (base MPMB)
        * *Na* : nombre d'antennes

    **Sortie**:
        * *Gamma* : matrice de covariance normalisée (base MPMB)
    """
    
    W_norm = normalize_MPMB_mat_PS(W,Na)
    E = power(np.diag(np.diag(W_norm.copy())),-0.5)    
    Gamma = E.dot(W_norm.dot(E))
    return Gamma
    
def normalize_MPMB_PS_Tebald(W,Na):
    """Normalisation de la matrice de covariance exprimée dans la base MPMB.
    
    La normalisation s\'effectue deux étapes:
    
        #. Stationnarité polarimetrique: *T1 = T2 = ... = 1/N sum Ti* 
        #. Normalisation selon chaquee canal et recepteur (tebaldini) : Ups_norm=(E-1/2 Ups E-1/2 avec E=diag(Ups))
                    
    **Entrées**:
        * *W* : matrice de covariance (base MPMB)
        * *Na* : nombre d'antennes
        
    **Sortie** :
        * *W_norm* : matrice de covariance normalisée (base MPMB)
        * *E* : diag(W_PS) avec W_PS, Ups_PS exprimé dans la base MPMB         
    """    
    
    T = np.zeros((Na,3,3),'complex64')
    Ups_PS = np.zeros((3*Na,3*Na),'complex64')
    Ups_norm = np.zeros((3*Na,3*Na),'complex64')
    Ups = MPMB_to_UPS(W)    
    for i in range(Na):        
        T[i][:,:] = Ups[i*3:i*3+3,i*3:i*3+3].copy()
    mean_T = T.mean(0)
    Ups_PS = Ups.copy()
    for i in range(Na):
        Ups_PS[i*3:i*3+3,i*3:i*3+3] = mean_T
        
    E = power(np.diag(np.diag(Ups_PS.copy())),-0.5)
    Ups_norm,E = blanch(Ups_PS)
    #Passage dans base MPMB
    W_norm = UPS_to_MPMB(Ups_norm)
    E = UPS_to_MPMB(E)
    return W_norm,E

def blanch(A):
    """Blanchiement de la matrice A
    
    A_blanc=(E-1/2 A E-1/2 avec E=diag(A))\n
    **Entrée** : A matrice à blanchir\n
    **Sorties** 
        * *A_blanc* : matrice blanchie
        * *EE* : EE=E^-1/2
    """
    EE = power(np.diag(np.diag(A)),-0.5)    
    A_blanc = EE.dot(A.dot(EE))
    return A_blanc,EE
    
def deblanch(W_blanc,E):
    """Deblanchi la matrice W_blanc
    E etant la mat diagonale contenant les 
    coeff diagonoaux (puissance -1/2) de la matrice 
    non blanchie"""
    
    F = npl.inv(E)
    return F.dot(W_blanc.dot(F))

def retranch_phig(zg,R_t,vec_kz):
    """Retranche la phase du sol aux matrices R_t.

    **Entrées** :
        * *zg* : altitude du sol
        * *R_t* : liste de matrices matrices structures (decomposition SKP). 
                  Contient les réponses interferométrique de chaque baseline  
        * *vec_kz* : vecteur contenant les kz dans l'ordre i<j
                     ex en dual-baseline vec_kz=kz12,kz13,kz23

    **Sortie** :
	* *R_t* : liste des matrices de structures avec phase du sol retranchée.
    """
    
    Na=mb.get_Na_from_Nb(len(vec_kz))
    print Na
    for p in range(len(R_t)):        
        for i in range(len(vec_kz)):
            idx=mb.get_idx_dble_from_idx_mono(i,Na)
            R_t[p][idx[0],idx[1]] = R_t[p][idx[0],idx[1]]*np.exp(-1j*vec_kz[i]*zg)
            R_t[p][idx[1],idx[0]] = R_t[p][idx[1],idx[0]]*np.exp(+1j*vec_kz[i]*zg)     
    return R_t

def retranch_phig_W(zg,W,vec_kz):
    """Retranche la phase du sol
    au niveau de la matrice de covariance (base MPMB)
    
    **Entrées** :
    
    * *zg* : altitude du sol
    * *W*  : mat de covariance (base MPMB)
    * *vec_kz* : vecteur contenant les kz dans l'ordre i<j\n
                 ex en dual-baseline vec_kz=kz12,kz13,kz23
                 
    """
    
    Na=mb.get_Na_from_Nb(len(vec_kz))
    Ups = MPMB_to_UPS(W)
    mat_rot = np.ones(Ups.shape,dtype='complex')
    
    p=0
    for i in range(Na-1):
        for j in range(i+1,Na):
            mat_rot[3*i:3*(i+1),3*j:3*(j+1)] = np.ones((3,3))*np.exp(-1j*vec_kz[p]*zg)
            mat_rot[3*j:3*(j+1),3*i:3*(i+1)] = np.ones((3,3))*np.exp(+1j*vec_kz[p]*zg)
            p = p+1    
    Ups_rot = Ups*mat_rot #multip terme à terme
    W_rot = UPS_to_MPMB(Ups_rot)
    return W_rot 
    
def sqrt_inverse(covar):
    """Retourne atemp verifiant atemp.dot(a_temp))=inv(covar)"""
    
    w,v=npl.eig(covar)
    atemp=np.sqrt(np.diag(1/w.real))
    atemp=v.dot(atemp.dot(v.T.conj()))
    return atemp
    
def sqrt_matrix(covar):   
    w,v=npl.eig(covar)
    atemp=np.diag(np.sqrt(w.real))
    atemp=v.dot(atemp.dot(v.T.conj()))
    return atemp    

def covar_inverse(covar):
    w,v=npl.eig(covar)
    atemp=(np.diag(1/w.real))
    atemp=v.dot(atemp.dot(v.T.conj()))
    return atemp
    
def polinsar_compute_omega12blanchi_basic(covar):
    """Calcule la matrice omega blanchi au sens de FF
    Attention la matrice doit être une 6x6
    
    **Entrée** : covar : matrice de covariance (base lexico)\n
    **Sortie** : covar : matrice de covariance normalisée (base lexico)\n
    """
    
    t11=covar[0:3,0:3]
    t22=covar[3:6,3:6]
    omega=covar[0:3,3:6]
    omega_blanchi=sqrt_inverse(t11).dot(omega.dot(sqrt_inverse(t22)))
    return omega_blanchi

def polinsar_compute_omega12blanchi(covar):
    """Calcule la matrice omega blanchi au sens de FF
    cela devrait marcher aussi pour la CP.\n
    Attention la matrice doit être une 6x6.

    **Entrée** : covar : matrice de covariance (base lexico)\n
    **Sortie** : covar : matrice de covariance normalisée (base lexico)\n    
    """
    temp=np.vsplit(covar,2)
    bloc=[np.hsplit(temp[0],2),np.hsplit(temp[1],2)]
    omega_blanchi=sqrt_inverse(bloc[0][0]).dot(bloc[0][1].dot(sqrt_inverse(bloc[1][1])))
    return omega_blanchi
    
def polinsar_estime_droite(omega):
    """Estimation des paramètres de la droite de cohérence 
    par la Méthode FF améliorée \n
    **Entrée** : omega : matrice de réponse interférométrique\n    
    **Sorties** : 
        * *theta* : angle d\'inclinaison par rapport à l\'axe horizontal
        * *d* : distance à l'origine
    """

    j=complex(0.,1.)
    pi_delta=omega-omega.T.conj()
    pi_sigma=omega+omega.T.conj()
    
    k_delta=pi_delta - pi_delta.trace()/3.*np.eye(3)
    k_sigma=pi_sigma - pi_sigma.trace()/3.*np.eye(3)
    
    #calcul des droites
    vc= -2*j*k_delta.dot(k_sigma).trace() + (j*(k_delta.dot(k_delta)+k_sigma.dot(k_sigma))).trace()
    theta1 = 0.5* np.arctan2(np.imag(j*vc), np.real(j*vc))
    theta2 = 0.5* np.arctan2(np.imag(-j*vc), np.real(-j*vc))
    d1 = (np.sin(theta1)*pi_sigma.trace() - j*np.cos(theta1)*pi_delta.trace())/6
    d2 = (np.sin(theta2)*pi_sigma.trace() - j*np.cos(theta2)*pi_delta.trace())/6
    
    #Calcl des critères 1 et 2
    n1 = np.cos(theta1)*pi_delta+j*np.sin(theta1)*pi_sigma-2*j*d1*np.eye(3)
    c1 = ((n1.dot(n1.T.conj())).trace()).real
    n2 = np.cos(theta2)*pi_delta + j*np.sin(theta2)*pi_sigma -2*j*d2*np.eye(3)
    c2=((n2.dot(n2.T.conj())).trace()).real
    d1=d1.real
    d2=d2.real
    
    if (c1 < c2):
        theta1,theta2=theta2,theta1
        d1,d2=d2,d1  
    return theta2,d2
    
def polinsar_ground_selection(covar,phi1,phi2,critere):
    """Suivant le critère choisi, on sélectionne la phase du sol entre phi1 et phi2
    Les critètres possibles sont hh-hv, hh-vv, hhmvv-hv"""
    
    j=complex(0,1)
    vect_droite=np.exp(j*phi2)-np.exp(j*phi1)
    if critere == 'hh-hv':
        canop = covar[1,4]/np.sqrt(covar[1,1]*covar[4,4]) # le hv est proche du haut de la canopee
        groun = covar[0,3]/np.sqrt(covar[0,0]*covar[3,3]) # le hh est proche du bas de la canopee
    elif critere == 'hh-vv':
        canop = covar[2,5]/np.sqrt(covar[2,2]*covar[5,5]) # le vv est proche du haut de la canopee
        groun = covar[0,3]/np.sqrt(covar[0,0]*covar[3,3]) # le hh est proche du bas de la canopee
       
    elif critere == 'hhmvv-hv':
        canop = covar[1,4]/np.sqrt(covar[1,1]*covar[4,4]) # le hv est proche du haut de la canopee
        temp=np.eye(6)
        temp[0,2] = -1
        temp[2,0] = 1
        temp[3,5] = -1
        temp[5,3] = 1
        covart=temp.dot(covar.dot(temp.T))
        groun = covart[0,3]/np.sqrt(covart[0,0]*covart[3,3]) # le hhmvv est proche du bas de la canopee
    vect_coh=canop-groun
    c1=(vect_droite*np.conj(vect_coh)).real
    if c1 < 0:
        phi1,phi2 = phi2,phi1
    return phi1,phi2
        
def polinsar_phase_intersection_cu(covar,theta2,d2,critere):
    #phi1,phi2 deux angles possible pour le sol

    phi1=np.pi/2-theta2+np.arccos(d2)
    phi2=np.pi/2-theta2-np.arccos(d2)
    
    if phi1 < 0:
        phi1 += 2*np.pi
    if (phi2 < 0):
        phi2=phi2+2*np.pi
    phi1,phi2=polinsar_ground_selection(covar,phi1,phi2,critere)
    return phi1,phi2

def polinsar_calcul_phig_psi(covar,critere='hh-hv'):
    """Effectue le calcul de la phase du sol et de ouverture angulaire
    à partir de la matrice de covariance
    Le critere est hh-hv,hh-vv,hhmvv-hv. Cette function retourne la
    phase du sol et le psi (phi2-phi)"""
    
    omega = polinsar_compute_omega12blanchi(covar)
    theta2,d2 = polinsar_estime_droite(omega)
    phi1,phi2 = polinsar_phase_intersection_cu(covar,theta2,d2,critere)
    return phi1,phi2-phi1

def polinsar_gamav(costeta,kz,extinction,hv):
    """Calcul la cohérence interférométrique du volume seul à partir de
    du costeta (cosinus de l'angle d'incidence, du kz, de l'extinction et du hv"""
    
    alpha=2*extinction/costeta
    a=np.exp(-alpha*hv)
    I1=(1-a)/alpha
    I2=(np.exp(complex(0.,1.)*kz*hv)-a)/(complex(0,1)*kz+alpha)
    return I2/I1
    
def polinsar_plot_cu(covar,title =' CU'):
    """Plot the cohérence region associated with the 6x6 covariance matrix 
    covar"""
    
    covarn = covar
    plt.figure(1)
    plt.axes(polar=True)
    #plt.title='test'
    T1=covarn[:3,:3]
    omega = covarn[:3,3:]
    #tracer plusieurs cohérences obtenues de manière alléatoire
    k=np.random.randn(20000,3) + 1j*np.random.randn(20000,3)
    power = ((k.dot(T1))*(np.conj(k))).sum(axis=1)
    interf = ((k.dot(omega))*(np.conj(k))).sum(axis=1)/power
    plt.plot(np.angle(interf),abs(interf),'c.')
    # tracer la droite de cohérence
    phig,psi = polinsar_calcul_phig_psi(covarn,'hh-hv') 
    plt.plot([phig,phig+psi],[1.,1.])    
    # tracer quelques points remarquables HH, HV, VV, phiG
    ghh = omega[0,0]/T1[0,0]
    plt.plot(np.angle(ghh),abs(ghh),'ro')
    ghv = omega[1,1]/T1[1,1]
    plt.plot(np.angle(ghv),abs(ghv),'go')
    gvv = omega[2,2]/T1[2,2]
    plt.plot(np.angle(gvv),abs(gvv),'bo')    
    plt.plot(phig,1.,'ko')
    plt.text(1.,1.2,title)        
    
def polinsar_plot_cu_orientation(covar,title=' CU'):
    """Plot the cohérence region associated with the 6x6 covariance matrix 
    covar - explore the orientation effect"""
    
    covarn = normalize_T1T2(covar)
    #plt.figure(num)
    p1=plt.figure()
    plt.axes(polar=True)
    T1=covarn[:3,:3]
    omega = covarn[:3,3:]
    # tracer plusieurs cohérences obtenues de manière alléatoire
    ia=[0,0,2,2]
    ib=[0,2,0,2]
    T2=covarn[ia,ib]
    T2.shape=(2,2)
    omega2=omega[ia,ib]
    omega2.shape=(2,2)  
    k=np.random.randn(500,2) + 1j*np.random.randn(500,2)
    power = ((k.dot(T2))*(np.conj(k))).sum(axis=1)
    interf = ((k.dot(omega2))*(np.conj(k))).sum(axis=1)/power
    plt.plot(np.angle(interf),abs(interf),'y.')

    T2=covarn[:2,:2]
    omega2=omega[:2,:2]  
    k=np.random.randn(500,2) + 1j*np.random.randn(500,2)
    power = ((k.dot(T2))*(np.conj(k))).sum(axis=1)
    interf = ((k.dot(omega2))*(np.conj(k))).sum(axis=1)/power
    plt.plot(np.angle(interf),abs(interf),'c.')

    T2=covarn[1:3,1:3]
    omega2=omega[1:3,1:3]  
    k=np.random.randn(500,2) + 1j*np.random.randn(500,2)
    power = ((k.dot(T2))*(np.conj(k))).sum(axis=1)
    interf = ((k.dot(omega2))*(np.conj(k))).sum(axis=1)/power
    plt.plot(np.angle(interf),abs(interf),'m.')
    
    
    # tracer la droite de cohérence
    phig,psi = polinsar_calcul_phig_psi(covarn,'hh-hv') 
    plt.plot([phig,phig+psi],[1.,1.])    
    # tracer quelques points remarquables HH, HV, VV, phiG
    ghh = omega[0,0]/T1[0,0]
    plt.plot(np.angle(ghh),abs(ghh),'ro')
    ghv = omega[1,1]/T1[1,1]
    plt.plot(np.angle(ghv),abs(ghv),'go')
    gvv = omega[2,2]/T1[2,2]
    plt.plot(np.angle(gvv),abs(gvv),'bo')
    omegab=polinsar_compute_omega12blanchi(covarn)
    plt.plot(phig,1.,'ko')
    plt.text(1.,1.2,title)
    
    #pylab.show()
    return   
    
def display_inversion_result(result):
    plt.figure()
    plt.imshow(result[0])
    plt.colorbar()
    plt.figure()
    plt.imshow(result[1])
    plt.colorbar()
    return
    
def calcul_matrix_derive(a,b,I1,I2,alpha,kz,hv,tvol,tground,omega):
    matrix_derive=np.zeros((6,6,20),dtype='complex')
    # dérivation par rapport à Tvol
    AA=[[1,0,0],[0,0,0],[0,0,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,0]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
                                     
    AA=[[0,0,0],[0,1,0],[0,0,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,1]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,0],[0,0,0],[0,0,1]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,2]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,1,0],[1,0,0],[0,0,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,3]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,1j,0],[-1j,0,0],[0,0,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,4]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,1],[0,0,0],[1,0,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,5]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,1j],[0,0,0],[-1j,0,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,6]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    
    AA=[[0,0,0],[0,0,1],[0,1,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,7]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,0],[0,0,1j],[0,-1j,0]]
    AA1=np.dot(I1,AA)
    AA2=np.dot(I2*b,AA)
    matrix_derive[:,:,8]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    # dérivation par rapport à Tground
    AA=[[1,0,0],[0,0,0],[0,0,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,9]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
                                     
    AA=[[0,0,0],[0,1,0],[0,0,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,10]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,0],[0,0,0],[0,0,1]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,11]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,1,0],[1,0,0],[0,0,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,12]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,1j,0],[-1j,0,0],[0,0,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,13]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,1],[0,0,0],[1,0,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,14]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,1j],[0,0,0],[-1j,0,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,15]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    
    AA=[[0,0,0],[0,0,1],[0,1,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,16]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    AA=[[0,0,0],[0,0,1j],[0,-1j,0]]
    AA1=np.dot(a,AA)
    AA2=np.dot(a*b,AA)
    matrix_derive[:,:,17]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    #Dérivation par rapport à zg
    AA1=np.zeros((3,3))
    AA2=np.dot(1j*kz,omega)
    matrix_derive[:,:,18]=np.vstack( (np.hstack( (AA1,AA2) ),
                                     np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    
    AA1 = np.dot(-alpha*a,tground)+np.dot(-a,tvol)
    xtamp = (1j*kz*np.exp(1j*kz*hv)+alpha*a) / (1j*kz+alpha)
    AA2 = np.dot(b*xtamp,tvol)+np.dot(-alpha*a*b,tground)
    matrix_derive[:,:,19]=np.vstack( (np.hstack( (AA1,AA2) ),
                                 np.hstack( (np.conj(AA2.transpose()),AA1) ) ))
    return matrix_derive
    
def sm_separation(W,Np,Na,Ki=2):
    
    """Separation de plusieurs mecanismes de diffusion à partir de la matrice 
    de covariance (base MPMB)
    
    Implémentation de la méthode de Telbaldini.
    
    **Entrées** : 
        * *W* : matrice de covariance des données (base MPMB) normalisée    .
        * *Ki* : nombre de mécanismes de rétro-diff (SM)
        * *Np* : nombre de polarisation (3 en FullPol)
        * *Na* : nombre d'antennes
        
    **Sorties** :
        * *R_t* : liste de matrices. Contient les matrices de strctures
        * *C_t* : liste de matrices. Contient les matrices de reponse polarimetriques
        * *G*   : matrice de covariance des données (base MPMB)
    """
    G = W.copy()#Sans normalisation. Normalisation censée être faite avant)"
    P_G = p_rearg(G,Np,Na) 
    
    
    """Attention : la fonction SVD renvoie 
    A = U S V.H
    Donc pour acceder aux vecteurs singuliers droits 
    il faut prendre le .H de la sortie si on veut
    que les vecteurs soit stockés en colonnes
    """
    
    mat_u,lmbda,mat_v_H = npl.svd(P_G)
    mat_v = mat_v_H.T.conj()
    #extraction des 2 premiers termes de la svd
    mat_u = mat_u[:,0:Ki]#Conserver les Ki premiers colonnnes
    mat_v = mat_v[:,0:Ki]#Conserver les Ki premiers colonnnes 
    lmbda = lmbda[0:Ki]
      
    U=[np.zeros((Np,Np))]*Ki
    V=[np.zeros((Na,Na))]*Ki
    C_t=[np.zeros((Np,Np))]*Ki
    R_t=[np.zeros((Na,Na))]*Ki
    
    #extraction des C_tilde et R_tilde
    for k in range(Ki): 
       U[k] = vec2mat(mat_u[:,k],3)#selection des Ki premiers vec sing gauche
       V[k] = vec2mat(mat_v[:,k],Na)#selection des Ki premiers vec sing droits
       C_t[k] = lmbda[k]*U[k]                       
       R_t[k] = V[k].conj()
       R_t_00 = R_t[k][0,0]
       R_t[k] = R_t[k]/R_t_00#normalisation     
       C_t[k] = C_t[k]*R_t_00    
       
    return R_t,C_t,G
    
def ground_selection_MB(R_t,interv_a,interv_b):
    """Selection du sol selon le criètre : \|coherence\| la plus elevée 
    
    **Entrées** : 
        * *R_t* : liste de matrice contenant les matrices de structures
                  du sol et du volume
        * *interv_a* : intervale de valuer possible pour a (cohérence du sol)
        * *interv_b* : intervale de valuer possible pour b (cohérence du volume)

    NB : un critère de def-positivité de matrice R_k et C_k permet d\'obtenir
    les valeurs de a (resp. de b) possibles pour calculer la matrice 
    de structure du sol (resp. du volume) à partir de la décomposition SKP 
    (cf Algebraic synthesis of forest scenarios[...]
    
    Attention : ne fonctionne seuleement qu'en MB 
    (en SB \|gammav\|=1 possible pour le sol ET le volume)"""
    
    vec_gamma = np.zeros((4,1),dtype='complex')    
    amin = interv_a[0][0]
    amax = interv_a[0][1]
    bmin = interv_b[0][0]
    bmax = interv_b[0][1]
    val = np.array([amin,amax,bmin,bmax])
    
    for i in range(4):
        vec_gamma[i] = val[i]*R_t[0][0,1]+(1-val[i])*R_t[1][0,1]
    idx_gmax = np.argmax(np.abs(vec_gamma))
    gmax =np.max(np.abs(vec_gamma))
    
    if idx_gmax == 0 or idx_gmax == 1:
        #la cohé max (signature du sol) est atteint pr des val de l'interv_a 
        # a <=> sol donc on ne change rien 
        interv_a_good =interv_a
        interv_b_good =interv_b
    elif idx_gmax == 2 or idx_gmax == 3:
        #la cohé max est atteint pr des val de l'interv_b 
        # b <=> vol Donc on inverse interva et interv b 
        #pour que la branche b coresp au vol et branche a au sol
        print 'grd selection : inversion interval'
        interv_a_good = interv_b
        interv_b_good = interv_a
    return interv_a_good,interv_b_good
    
def value_R_C(R_t,C_t,a,b): 
        """Calcul des matrices de structures (R_t) et de réponses polarimétriques
        C_k à partir des valeurs a et b
        
        Attention : R_t,C_t doivent être sous forme diagonale. Voir 
        Tebaldini, Algebraic synthesis of forest scenarios[...]
        N.B : W = R_t[0]oC_t[0] + R_t[1]oC_t[1] (o : prod de kronecker)
        
        **Entrées** : 
            * *R_t* : liste des matrices de structure
            * *C_t* : liste des réponses polarimétrique
            * *a,b* : scalaires fixant la cohérence du sol (resp. du volume)
        
        **Sorties** :
            * *vap* : valeurs diagonales de R1,R2,C1 et C2 dans un seul vecteur
            * *R1*, *R2* : matrice de struce du sol et du volume
            * *C1*,*C2* : reponse polarimétique du sol et du volume
        """
        
        R1 = a*R_t[0]+(1-a)*R_t[1]
        R2 = b*R_t[0]+(1-b)*R_t[1]
        C1 = 1/(a-b)*((1-b)*C_t[0]-b*C_t[1])
        C2 = 1/(a-b)*(-(1-a)*C_t[0]+a*C_t[1])
        #Renvoi des valeurs diagonales dans un seul vecteur      
        
        vap_a = np.hstack((np.diag(R1),np.diag(C2)))
        vap_b = np.hstack((np.diag(R2),np.diag(C1)))
        vap = np.hstack((vap_a,vap_b))
        return vap,R1,R2,C1,C2

def taille_intervb(R_t,interv_b): 
    """Renvoie la taille de l'intervalle des gammav possibles
    
    Si le nbre de baseline est superieur à 1 interv size
    est un vecteur contenant la taille de l'intervalle sur 
    chaque baseline.
    
    **Entrées** : 
            * *R_t* : liste des matrices de structure
            * *interv_b* : liste contenant les bornes inf et sup de l'interval de
                            *b* possibles (cohérence du volume)
        
    **Sorties** :
            * *interv_size* : taille des intervalles de cohérences possibles
    """
    
    Na = R_t[0].shape[0]
    interv_size = np.zeros(Na)
    Rv0= interv_b[0][0]*R_t[0]+(1-interv_b[0][0])*R_t[1]
    Rv1= interv_b[0][1]*R_t[0]+(1-interv_b[0][1])*R_t[1]
    vec_gamma0 = gamma_from_Rv(Rv0) #ensemble des cohé b=bmin
    vec_gamma1 = gamma_from_Rv(Rv1) #ensemble de cohéb=bmax  
    interv_size=np.abs(vec_gamma0-vec_gamma1)
    return interv_size
    
def gamma_a_b(R_t,a,b,ant1=0,ant2=1):
    """Renvoie la coherence de chaque R_k pour une valeur du couple 
    (a,b) (Ki=2)"""
    
    gamma1 = a*R_t[0][ant1,ant2]+(1-a)*R_t[1][ant1,ant2]
    gamma2 = b*R_t[0][ant1,ant2]+(1-b)*R_t[1][ant1,ant2]
    return gamma1,gamma2
    
def rac_def_pos(R_t,a,b):
    """Calcul les racines du polyme pdp pour 
    une valeur de gamma_v définie par b"""
    
    Na = R_t[0].shape[0]
    Nb_baseline = int(Na*(Na-1)/2)
    gamma_Rv = np.zeros(Nb_baseline,dtype='complex')   
    idx_g=0     
    for i in range(Na-1):
        for j in range(i+1,Na):
            #print i,j,idx_g,b*R_t[0][i,j]+(1-b)*R_t[1][i,j] #debug
            gamma_Rv[idx_g] = b*R_t[0][i,j]+(1-b)*R_t[1][i,j]
            idx_g +=1
    ratio = gamma_Rv[1]/(gamma_Rv[0]**2) #g13/g12^2
    z,alpha,beta,r1,r2,coeffa = pol_pdp(ratio)
    
    return gamma_Rv,ratio,r1,r2,a
    
def pol_pdp(ratio):
    """Renvoie les racine du polynomes verfié par \|gamma12\|² 
    dans le cas de 3 Baselines et 
    les hypothèses *g12 = g12g12*alpha*exp(i*beta)*
    et *g23 = g12* """ 
    
    z= 1-ratio
    alpha = np.abs(ratio)
    beta = np.angle(ratio)
    r1 = (np.abs(z)-1)/(alpha*(alpha-2*np.cos(beta)))
    r2 = -(np.abs(z)+1)/(alpha*(alpha-2*np.cos(beta)))
    coeffa = -alpha*(alpha-2*np.cos(beta))
    return z,alpha,beta,r1,r2,coeffa
       
def interv_possible(alpha,mat_cond,Na):    
    """Renvoi les intervalles pour a et b dont les valeurs donnent des
    matrices R_t et C_t définies positives
    
    **Entrées** 
        * *alpha* : racines des equations de positivité    
        * *mat_cond* : matrice binaire dont l'état correspond à la validation
                       de la condition de positivité (1 si vraie 0 sinon)
                       
    **Sorties** : 
        * *interv_a*, *interv_b* : intervalles possibles pour a et b              
    """
    
    index = zip(*np.where(mat_cond==1))        
    #Construction de la matrice contenant les valeurs des intervales
    #on a (Na + 3) 'alpha' (Na par les mat R_k, 3 par les mat C_k) 
    #interv possible ]-oo;a0],[a0,a1],..,[a_Na+1,a_Na+2],[a_Na+2,+oo]
    # soit Na+4 intervalle
    """Préferons la redondance à l'errance"""
    mat_interv = np.ndarray((Na+4,Na+4),dtype=object) #array contenant des tuples
    
    interv_a=[]
    interv_b=[]
    #les quatres coins particulier 
    mat_interv[0,0] = ((-np.inf,alpha[0]),(-np.inf,alpha[0]))
    mat_interv[0,-1] = ((-np.inf,alpha[0]),(alpha[-1],np.inf))
    mat_interv[-1,0] = ((alpha[-1],np.inf),(-np.inf,alpha[0]))
    mat_interv[-1,-1] = ((alpha[-1],np.inf),(alpha[-1],np.inf))
    #Remplissage des lignes et colonnes exterieures
    for i in range(1,Na+3):# indice de 1 à Na+2 =>Na+2 elements        
        #parcours des lignes exterieures
        mat_interv[i,0] = ((alpha[i-1],alpha[i]),(-np.inf,alpha[0]))
        mat_interv[i,-1] = ((alpha[i-1],alpha[i]),(alpha[-1],np.inf))
        #parcours des colonnes exterieures    
        mat_interv[0,i] = ((-np.inf,alpha[0]),(alpha[i-1],alpha[i]))
        mat_interv[-1,i] = ((alpha[-1],np.inf,),(alpha[i-1],alpha[i])) 
    #Carré interieur
    for i in range(1,Na+3):
        for j in range(1,Na+3):
            mat_interv[i,j]=((alpha[i-1],alpha[i]),(alpha[j-1],alpha[j]))
            
    #SUGGESTIONS DE MODIF : AU LIEUR DE PRENDRE LA MOITIE ARBITRAITEMENT 
    # PRENDRE LES a>b ou a<b             
    if len(index)%2==0 and len(index)>1:                    
        for i in range(len(index)//2): #On en la moitié (mat_interv symetrique)
            interv_a.append(mat_interv[index[i]][0])
            interv_b.append(mat_interv[index[i]][1])
            
    elif len(index)==1:
        interv_a.append(mat_interv[index[0]][0])
        interv_b.append(mat_interv[index[0]][1])

    return interv_a,interv_b
    
def search_space_definition(R_t,C_t,Na):
    """Permet de déterminer l'interval des valeurs possibles de a et b
    en fonction des valeurs des valeurs propres R_t_diag et C_t_diag.
    
    **Entrées** : 
        * *R_t* : liste des matrices de structure
        * *C_t* : liste des réponses polarimétrique
        * *Na* : nombre d'antennes
      
    **Sorties** : 
        * *interv_a,interv_b* : intervals possibles pour les val. de a et b
        * *mat_cond* : matrice contenant 1 si la condition de positivité est 
                       vérifiée, 0 sinon.
        * *alpha* : liste des valeurs ou les équations de positivité s'annulent    
    """  
    
    #Nombre de SM
    Ki=2   
    R_t_diag=[None]*Ki # Ki vecteurs (de dimensiosn Na) contenant les vap        
    C_t_diag=[None]*Ki 
    alpha_C = np.zeros((3,1))
    alpha_R = np.zeros((Na,1))    
    # 'Diagonalisation 'commun
    R_t_diag[0],R_t_diag[1],_,_ = ejd(R_t[0],R_t[1]) #ndice ~ k ~ num du SM 
    C_t_diag[0],C_t_diag[1],_,_ = ejd(C_t[0],C_t[1])  
    #definition des bords des intervalles possibles
    
    for i in range(Na):        
        alpha_R[i]=np.real(R_t_diag[1][i,i]\
                        /(R_t_diag[1][i,i]-R_t_diag[0][i,i]))
    for i in range(3):        
        alpha_C[i] = np.real(C_t_diag[0][i,i]\
                        /(C_t_diag[0][i,i]+C_t_diag[1][i,i]))
                        
    #on classe les alpha relatif à a et b (ce sont les même pour a ou b)    
    alpha = np.vstack((alpha_R,alpha_C))
    alpha= np.real(alpha)
    alpha.sort(0) #0 pour classer selon la premier dimension c.a.d les lignes   
    
    interv_a = [] #contient les intervales où les contraintes de positiv sont 
    interv_b = [] #valides. (a_min,a_max)
    
    Sa=1 #seuil 
    a_test_debut= [alpha[0]-Sa]
    a_test_mil = [(alpha[i]+alpha[i+1])/2 for i in range(alpha.size-1)]
    a_test_fin = [alpha[-1]+Sa] 
    a_test = a_test_debut+a_test_mil+a_test_fin
    
    beta = 0.75
    Sb = 2 #seuil
    b_test_debut= [alpha[0]-Sb]
    b_test_mil = [(beta*alpha[i]+(1-beta)*alpha[i+1]) for i in range(alpha.size-1)]
    b_test_fin = [alpha[-1]+Sb] 
    b_test = b_test_debut+b_test_mil+b_test_fin
    
    mat_cond = np.zeros((len(a_test),len(b_test)))
    mat_cond_vap = np.zeros((len(a_test),len(b_test)))

    for i,a in enumerate(a_test):
        for j,b in enumerate(b_test):             
            mat_cond[i,j] = positivity_condition(R_t_diag,C_t_diag,a,b)
           
    #Verification: mat_cond doit être symetrique 
    if np.sum(mat_cond-mat_cond.T != 0):
        print 'Attention matrice de conditions non symetrique !!'
    else:
        index = zip(*np.where(mat_cond==1))        
    #Construction de la matrice contenant les valeurs des intervales
    #on a (Na + 3) 'alpha' (Na par les mat R_k, 3 par les mat C_k) 
    #interv possible ]-oo;a0],[a0,a1],..,[a_Na+1,a_Na+2],[a_Na+2,+oo]
    # soit Na+4
    interv_a,interv_b = interv_possible(alpha,mat_cond,Na)
    if np.sum(mat_cond.ravel()) == 0:
        print 'search_space_definition: Aucun interval de total positivité trouvé '
        #interv_a.append([0,0])
        #interv_b.append([0,0])
        
    return interv_a,interv_b,mat_cond,alpha


def search_space_definition_rob(R_t,C_t,Na,Ki=2):
    """Permet de définir l'interaval des valeurs possibles de a et b
    en fonction des valeurs des valeurs propres R_t_diag et C_t_diag.
    Version améliorée de search_space_definition pour gérer le cas 
    où aucun interval n'est trouvé

    **Entrées** : 
        * *R_t* : liste des matrices de structure
        * *C_t* : liste des réponses polarimétrique
        * *Na* : nombre d'antennes
      
    **Sorties** : 
        * *interv_a,interv_b* : intervals possibles pour les val. de a et b
        * *mat_cond* : matrice contenant 1 si la condition de positivité est 
                       vérifiée, 0 sinon.
        * *alpha* : liste des valeurs ou les équations de positivité s'annulent    
    """      
    
    #Nombre de SM
    Ki=2   
    R_t_diag=[None]*Ki # Ki vecteurs (de dimension Na) contenant les vap        
    C_t_diag=[None]*Ki 
    alpha_C = np.zeros(3)
    alpha_R = np.zeros(Na)    
    # 'Diagonalisation 'commun
    R_t_diag[0],R_t_diag[1],_,_ = ejd(R_t[0],R_t[1]) #ndice ~ k ~ num du SM 
    C_t_diag[0],C_t_diag[1],_,_ = ejd(C_t[0],C_t[1])  
    #definition des bords des intervalles possibles    
    for i in range(Na):        
        alpha_R[i]=np.real(R_t_diag[1][i,i]\
                        /(R_t_diag[1][i,i]-R_t_diag[0][i,i]))
    for i in range(3):        
        alpha_C[i] = np.real(C_t_diag[0][i,i]\
                        /(C_t_diag[0][i,i]+C_t_diag[1][i,i]))
    #alpha_C_b_1=alpha_C_a_1 
    #alpha_C_b_2=alpha_C_a_2
    
    #on classe les alpha relatif à a et b (ce sont les même pour a ou b)    
    alpha = np.hstack((alpha_R,alpha_C))
    alpha= np.real(alpha)
    alpha.sort()   
    
    interv_a = [] #contient les intervales où les contraintes de positiv sont 
    interv_b = [] #valides. (a_min,a_max)
    
    Sa=1 #seuil 
    a_test_debut= [alpha[0]-Sa]
    a_test_mil = [(alpha[i]+alpha[i+1])/2 for i in range(alpha.size-1)]
    a_test_fin = [alpha[-1]+Sa] 
    a_test = a_test_debut+a_test_mil+a_test_fin
    
    beta = 0.75
    Sb = Sa+1 #seuil
    b_test_debut= [alpha[0]-Sb]
    b_test_mil = [(beta*alpha[i]+(1-beta)*alpha[i+1]) for i in range(alpha.size-1)]
    b_test_fin = [alpha[-1]+Sb] 
    b_test = b_test_debut+b_test_mil+b_test_fin
    
    mat_cond = np.zeros((len(a_test),len(b_test)))
    mat_value = np.ndarray((len(a_test),len(b_test)),dtype=object)
    mat_neg = np.ndarray((len(a_test),len(b_test)),dtype=object)
    mat_max_abs_neg = np.ndarray((len(a_test),len(b_test)))
    # chaque case de mat_value contient un vecteur de dim Na+4 valeures
    #soit la valeur des vap des matrices R_k et C_k au milieu des intervalles
    # ]-oo,alpha1] [alpha2,alpha3] .. [alpha_Na+3] [alpha_Na+3,+oo]
        
    for i,a in enumerate(a_test):
        for j,b in enumerate(b_test):             
            mat_cond[i,j] = positivity_condition(R_t_diag,C_t_diag,a,b)
            mat_value[i,j] = np.real(value_R_C(R_t_diag,C_t_diag,a,b)[0])
    #Verification: mat_cond doit être symetrique 
    if np.sum(mat_cond-mat_cond.T != 0):
        print 'Attention matrice de conditions non symetrique !!'

    index = zip(*np.where(mat_cond==1))      
    #Construction de la matrice contenant les valeurs des intervales
    #on a (Na + 3) 'alpha' (Na par les mat R_k, 3 par les mat C_k) 
    #interv possible ]-oo;a0],[a0,a1],..,[a_Na+1,a_Na+2],[a_Na+2,+oo]
    # soit Na+4
    if np.sum(mat_cond.ravel()) == 0:
        print 'search_space_definition_rob: Aucun interval de total positivité trouvé '            
        Zer = np.zeros((1,mat_value[0,0].size))
        for i in range(len(a_test)):
            for j in range(len(b_test)):                             
                #extraction des négatif
                mat_neg[i,j] = np.min(np.vstack((Zer,mat_value[i,j])),axis=0) 
                mat_max_abs_neg[i,j] = np.max(np.abs(mat_neg[i,j]))
                
        flat_idx = mat_max_abs_neg.argmin()
        idx = np.unravel_index(flat_idx,mat_neg.shape)
        idx_transp=(idx[1],idx[0])
        mat_cond[idx] = 1
        mat_cond[idx_transp] = 1        
        
        interv_a,interv_b = interv_possible(alpha,mat_cond,Na)
        print 'interv_a propose',interv_a,'intervb propose',interv_b
        #print '==============='        
        if interv_a==[] or interv_b==[]:
            #print 'TOjours aucun interval selectionne!!'
            pdb.set_trace()
        approx = 1
    else:       
        interv_a,interv_b = interv_possible(alpha,mat_cond,Na)
        approx = 0
            
    return interv_a,interv_b,mat_cond,alpha,approx


def search_space_definition_brute(R_t,C_t,plot = 1):
    """Recherche des intervalle de valeurs pour a et b où la 
    condition de positivité est vérifiée. 
    
    Méthode brute force (pour valider les méthodes plus 
    fines mais seuls les test sur intervals interv_a et 
    interv_b suffisent à priori): test aveugle sur a et b

    **Entrées** : 
        * *R_t* : liste des matrices de structure
        * *C_t* : liste des réponses polarimétrique
        * *Na* : nombre d'antennes
      
    **Sorties** : 
        * *interv_a,interv_b* : intervals possibles pour les val. de a et b
        * *mat_cond* : matrice contenant 1 si la condition de positivité est 
                       vérifiée, 0 sinon.
        * *alpha* : liste des valeurs ou les équations de positivité s'annulent    
    """      
    
    step=0.001
    #pour a
    a_valable = []
    b_valable = []    
    a_test=np.arange(0,1.2,step) 
    b_test=np.arange(0,1.2,step) 
    gamma_valable_a = []
    gamma_valable_b = []
    Cond = np.zeros((a_test.size,b_test.size))
    for i,a in enumerate(a_test):    
        for j,b in enumerate(b_test):
            if a != b: 
                R1 = a*R_t[0]+(1-a)*R_t[1]
                R2 = b*R_t[0]+(1-b)*R_t[1]
                C1 = ((1-b)*C_t[0]-b*C_t[1])/(a-b)                
                C2 = (-(1-a)*C_t[0]+a*C_t[1])/(a-b)
                
                eig_R1,_ = npl.eig(R1)
                eig_R2,_ = npl.eig(R2)
                eig_C1,_ = npl.eig(C1)
                eig_C2,_ = npl.eig(C2)
                
                #concatenation de toutes les val propres
                eigen_a = np.hstack((eig_R1,eig_C2))
                eigen_b = np.hstack((eig_R2,eig_C1))
                eigen = np.hstack((eigen_a,eigen_b))
                
                if (mat.find(eigen<0).size == 0):#Si aucun élément n'est strict négatif            
                    Cond[i,j] = 1
                else:
                    Cond[i,j] = 0
    return Cond
    
   
def b_true(param):
    """Renvoie le bvrai à partir d'un objet param, 
    si l'on connait les paramètres du modèle
    
    Méthode mimisant ||Rv-Rv(b)||² avec Rv matrice de structure du volume 
    obtenue par le modèle RVoG
    
    **Entrée** : 
        * *R_t* : liste des matrices de structure 
        * *param* : object de type param_rvog
        
    **Sortie** : 
        * *b_vrai* : valeur vraie du paramètre *b*"""
    
    Np=3
    W = UPS_to_MPMB(param.get_upsilon_gt())
    W_norm,_ = normalize_MPMB_PS_Tebald(W,param.Na)
    R_t,_,_ = sm_separation(W_norm,Np,param.Na)
    Rv = param.get_Rv()
    Rt1 = R_t[0]
    Rt2 = R_t[1]    
    btrue= -np.trace((Rv-Rt2).T.conj().dot(Rt2-Rt1))/\
            (np.trace((Rt2-Rt1).dot(Rt2-Rt1)))
    return np.real(btrue)    
    
def a_true(R_t,param):
    """Renvoie le avrai si l'on connait les paramètres du modèle
   
    Méthode mimisant ||Rg-Rv(b)||² avec Rg matrice de structure du sol
    obtenue par le modèle RVoG
    
    **Entrée** : 
        * *R_t* : liste des matrices de structure 
        * *param* : object de type param_rvog
        
    **Sortie** : 
        * *a_vrai* : valeur vraie du paramètre *a*"""
        
    Rg = param.get_Rg()
    Rt1=R_t[0]
    Rt2=R_t[1]
    atrue = -np.trace((Rg-Rt2).T.conj().dot(Rt2-Rt1))/\
              (np.trace((Rt2-Rt1).dot(Rt2-Rt1)))
    return np.real(atrue)
    
def positivity_condition_R(R_t,a,b):
    """Evalue la condition de positivité seulement pour les matrices R1 R2 
    dans (25) de Tebaldini en fonction de a et b 
    
    Attention : les R_t doivent être sous forme diagonale (au sens
    de la diagonalisation conjointe) cf fonction ejd
    
    **Entrée** : 
        * *R_t* : liste des matrices de structure 
        * *a*, *b* : scalaires reliées aux cohérences du sol (resp. du volume).
        
    **Sortie** : 
        * *pos_condition* : booleen. =1 si condition de positivité vraie,0 sinon.
    """
    
    if a==b :
        print 'Erreur a=b !'
        return 0
    
    else:        
        R1 = a*R_t[0]+(1-a)*R_t[1]
        R2 = b*R_t[0]+(1-b)*R_t[1]
    
        Cond_a = np.diag(R1)
        Cond_b = np.diag(((R2),np.diag(C1)))
        Cond = np.hstack((Cond_a,Cond_b))        
        
        if(mat.find(Cond<0).size == 0):#Si aucun élément n'est négatif
            pos_condition = True 
        else:
            pos_condition = False
        return pos_condition

def positivity_condition(R_t,C_t,a,b):
    """Verifie la condition de positivité pour l'ensemble des matrices R1 R2 C1 et C2 
    dans (25) de Tebaldini pour a et b.
   
    Attention : les R_t doivent être sous forme diagonale (au sens
    de la diagonalisation conjointe) cf fonction ejd
    
    **Entrée** : 
        * *R_t* : liste des matrices de structure 
        * *R_t* : liste des matrices de réponses polarimétriques        
        * *a*, *b* : scalaires reliées aux cohérences du sol (resp. du volume).
        
    **Sortie** : 
        * *pos_condition* : booleen. =1 si condition de positivité vraie,0 sinon.
    """
    
    if a==b :
        print 'Erreur a=b !'
        return 0    
    else:        
        R1 = a*R_t[0]+(1-a)*R_t[1]
        R2 = b*R_t[0]+(1-b)*R_t[1]
        C1 = 1/(a-b)*((1-b)*C_t[0]-b*C_t[1])
        C2 = 1/(a-b)*(-(1-a)*C_t[0]+a*C_t[1])
        
        Cond_a = np.hstack((np.diag(R1),np.diag(C2)))
        Cond_b = np.hstack((np.diag(R2),np.diag(C1)))
        Cond = np.hstack((Cond_a,Cond_b))        
        
        if(mat.find(Cond<0).size == 0):#Si aucun élément n'est négatif
            pos_condition = True 
        else:
            pos_condition = False
        return pos_condition
       
def bounded_condition(R_t,a):
    """Verifie que la coherence interferometrique d'un SM donné
    est inferieure à 1.
    
    
    **Entrée**: 
        * *R_t* : listes des matrices de structures
        * *a*   : valeur de *a* dans la condition
        
    **Sorties**:
        * *Cond* : 1 si la condition est verifiée, 0 sinon
        * *gamm* : valeure de la cohérence 
    """
    
    R = a*R_t[0]+(1-a)*R_t[1]
    gamm = R[0,1]
    if(np.abs(gamm)<=1):
        Cond=True
    else:
        Cond=False
    return(Cond,gamm)
        
def p_rearg(A,Np,Na):
    """Operateur de réarrangement. (cf annexe dans 
    Tebaldini, Algebraic synthesis of forest scenarios[...])
    
    **Entrée** : 
        * *A* :  matrix par bloc de NpxNp blocs de taille NaxNa
        * *Na* : nombre d'acquisitions (antennes)
        * *Np* : nombre de polarisations (3 en full polar)
        
    **Sortie** :
        * *A_rearg* : matrice reearrangée
    """
    
    A_j = np.zeros(Na*Na,dtype=complex)
    
    for j in range(Np):
        for i in range(Np):
            A_j=np.vstack((A_j,A[i*Na:(i+1)*Na,j*Na:(j+1)*Na].T.ravel())) 
    A_rearg = A_j[1:,:] 
    return A_rearg
    
def inv_p_rearg(P_A,Np,Na):
    """Operateur inverse de p_rearg
    
    **Entrée** : 
    * *Na* : nombre d'acquisitions (antennes)
    * *Np* : nombre de polarisations 
    * *A*  : matrice de sortie """ 
    
    A = np.zeros((Na*Np,Na*Np),dtype=complex)
    bloc = np.zeros((Na**2,1),dtype=complex)
    j = int()
    for j in range(Np**2):        
        bloc = P_A[j,:]
        A[Na*(j%Np):Na*(j%Np)+Na,Na*(j//Np):Na*(j//Np)+Na]=\
                                                vec2mat(bloc,Na)
    return A
    
def power(A,n):
    """Renvoie la puissance n d'une matrice (array)
    (méthode des valeurs propres)"""  
    
    D,P=npl.eig(A)
    D=D**n
    D=np.diag(D)
    A_pow=P.dot(D.dot(npl.inv(P)))
    return A_pow
    
    
def gamma_from_Rv(Rv):
    """Extraction des gamma de la matrice de structure Rv
    les gamma sont situéss au cordonées i<j"""
    Na = Rv.shape[0]
    vec_gamma=np.zeros((np.floor(Na*(Na-1)/2)),dtype='complex')
    p=0
    for i in range(Na-1):
        for j in range(i+1,Na):
            vec_gamma[p]=Rv[i,j]
            p +=1
    return vec_gamma
    
def mat2vec(A):
    """Matrix to vector operator. Transforme la matrice en vecteur en 
    concatenant les colonnes de A"""
    a = A.ravel()
    return a     

def vec2mat(a,N):
    """vector to matrix operator. Transforme le vecteur *a* 
    en matrice *NxN* en \'reshapant\' selon les colonnes
    
    **Entrée** : 
        *a* vector to reshape
        *N* size of the square matrix
        
    **Sortie** : 
	* *A* : matrice dont les colonnes sont issues du vec. a.
    """
    A = a.reshape(N,N,order='F')
    return A

def ejd(Q1,Q2):
    """Exact joint diagonalisation: diagonalize Q1 and Q2 in a common basis
       extract from "On Using Exact Joint Diagonalization for Noniterative 
       Approximate Joint Diagonalization" by Arie Yeredor
    
    **Entrée** : 
    	* *Q1,Q2* : matrix to diagonalise 
    
    **Sorties** : 
        * *D1* : diag matrix containing eigenvalues of Q1
        * *D2* : diag matrix containing eigenvalues of Q2
        * *A* : matrix of change of basis
        * *LAMBDA_mat* : eigenvalues of Q1*inv(Q2)
    """
    
    R=np.dot(Q1,npl.inv(Q2))
    LAMBDA_vec,A = npl.eig(R)
    LAMBDA_mat=np.diag(LAMBDA_vec)#mise sous forme de matrice
    A_H=np.conj(A.T)
    D2=npl.inv(A).dot(Q2.dot(npl.inv(A_H)))        
    D1=np.dot(LAMBDA_mat,D2)       
    return D1,D2,A,LAMBDA_mat

def ejd2(A,B):
    """Exact joint diagonalisation: diagonalize Q1 and Q2 in a common basis
       extract from "On Using Exact Joint Diagonalization for Noniterative 
       Approximate Joint Diagonalization" by Arie Yeredor

    N.B : version améliorée de ejd pour selectionner entre 
    la matrice la moins singulière entre Q1 et Q2 
    
    **Entrée** : 
    * *A,B* : matrix to diagonalise 
    
    **Sorties** : 
    * *D1* : diag matrix containing eigenvalues of Q1
    * *D2* : diag matrix containing eigenvalues of Q2
    * *A* : matrix of change of basis
    * *LAMBDA_mat* : eigenvalues of Q1*inv(Q2)
    """
    #On selection Q2 comme la matrice la mieux conditionnée
    #(la moins singulière)
    
    list_mat = [A,B]
    condi = [npl.cond(A),npl.cond(B)]
    Q1 = list_mat[np.argmax(condi)]
    Q2 = list_mat[np.argmin(condi)]
    
    R= Q1.dot(npl.inv(Q2))
    LAMBDA_vec,A = npl.eig(R)
    LAMBDA_mat=np.diag(LAMBDA_vec)#mise sous forme de matrice
    A_H=np.conj(A.T)
    
    D1=npl.inv(A).dot(Q1.dot(npl.inv(A_H)))            
    D2=npl.inv(A).dot(Q2.dot(npl.inv(A_H)))        
    return D1,D2,A,LAMBDA_mat


def idx_teb_to_idx_std(idx_teb,Na):
    """Renvoie le coordonées du vecteur en coord tebaldini à
    partir des coord lexicographiques.
    
    Effectue le changnement de base suivant :\n
    (hh1,hh2,...,hhNa,hv1,hv2,...,hvNa,vv1,vv2,..,vvNa)->(hh1,hv1,vv1,hh2,hv2,vv2,...,hhNa,hvNa,vvNa).
    ex idx_teb_to_idx_std(,Na):
    
    **Entrée** : 
        * *idx_teb* : index du vec. de données MPMB
        * *Na* : nombre d'antennes
        
    **Sortie** :
        * *idx_std* : index correspondant en coordonnées lexico. 
    """
    
    #matrice de changement de var
    P=np.zeros((Na*3,Na*3))
    vec_in=np.zeros(3*Na)
    vec_in[idx_teb]=1
    for i in range(3):
        for j in range(Na):
            P[i+3*j,i*Na+j]=1
    vec_out=P.dot(vec_in)
    idx_std = np.where(vec_out==1)[0][0]
    return idx_std
    
def MPMB_to_UPS(W):
    """Renvoie la matrice de covariance en coordonnées standard PolInSAR
    à partir de la matrice de covariance en coordonnées MPMB.
    
    coordoonnées standards : (hh1,hv1,vv1,hh2,hv2,vv2,...,hhNa,hvNa,vvNa)
    coordonnées MPMB: (hh1,hh2,...,hhNa,hv1,hv2,...,hvNa,vv1,vv2,...,vvNa)
    
    **Entrée** : 
        * *W* : matrice de covariance en coord MPMB taille (9xNa²)
        
    **Sortie**: 
        * *Ups* : matrice de covariance en coord lexicographique.
    """    
    
    Na = int(np.shape(W)[0]/3)
    #matrice de changement de var
    P=np.zeros((Na*3,Na*3))
    for i in range(3):
        for j in range(Na):
            P[i+3*j,i*Na+j]=1
            
    Ups=P.dot(W.dot(P.T.conj()))
    return Ups

def UPS_to_MPMB(Ups):
    """Renvoie la matrice de covariance en coordonnées MPMB
    à partir de la matrice de covariance en coordonnées lexicographiqe.
    
    coordonnées MPMB: (hh1,hh2,...,hhNa,hv1,hv2,...,hvNa,vv1,vv2,...,vvNa)
    coordoonnées standards : (hh1,hv1,vv1,hh2,hv2,vv2,...,hhNa,hvNa,vvNa)
        
    **Entrée** : 
        * *Ups* : matrice de covariance en coord lexicographique.    
        
    **Sortie**: 
        * *W* : matrice de covariance en coord MPMB taille (9xNa²)
    """    

    Na = int(np.shape(Ups)[0]/3)
    #matrice de changement de var
    P=np.zeros((Na*3,Na*3))
    for i in range(3):
        for j in range(Na):
            P[i+3*j,i*Na+j]=1
            
    P_inv=npl.inv(P)
    W=P_inv.dot(Ups.dot(P_inv.T.conj()))
    return W
    
def UPS_MB_to_SB(Ups_MB,ant1,ant2,Na):    
    """ Renvoie la matrice de covariance PolInSAR single baseline
        pour une baseline determinée par les indices d'antenne ant1
        et ant2.
        
        **Entrées**:
        
        * *Ups_MB* : matrice de covariance multi-baseline (MB)
        * *ant1,ant2* : numeros des antennes 
        * *Na* : nombre d'antennes
        
        **Sortie** : *Ups_SB* : matrice de covariance single-baseline (SB)
    """
    
    if (ant1>Na or ant2>Na):        
        print "Attention! : numero d'antenne trop grand"
        print 'ant1 =',ant1,' ant2 =',ant2,' Na=',Na
        
    #conversion numero antenne en index
    ant1 = ant1-1
    ant2 = ant2-1    
    if (ant1 == ant2):        
        print "Attention! : même numéro d'antenne => pas de matrice interferometrique"
    
    #Extraction matrices T   
    T_ant1 = Ups_MB[ant1*3:ant1*3+3,ant1*3:ant1*3+3]
    T_ant2 = Ups_MB[ant2*3:ant2*3+3,ant2*3:ant2*3+3]
    #Extraction matrice Omega_ant1,ant2    
    Omega= Ups_MB[ant1*3:ant1*3+3,ant2*3:ant2*3+3]
    Omega_H = Omega.T.conj()
    Ups_SB= np.vstack((np.hstack((T_ant1,Omega)),np.hstack((Omega_H,T_ant2))))
    return Ups_SB

def MPMB_to_SB(MPMB,ant1,ant2,Na):
    """ Renvoie la matrice de W_k (matrice de covariance des données MPMB {tebaldini})
        en single baseline pour une baseline determinée par les indices d'antenne ant1
        et ant2
        
        **Entrées**:
        
        * *MPMB* : matrice de covariance MB en coord. MPMB
        * *ant1,ant2* : numeros des antennes 
        * *Na* : nombre d'antennes
        
        **Sortie** : *Ups_SB* : matrice de covariance SB en coord. MPMB
                
    """
    
    Na_SB=2
    #conversion numero antenne en index
    ant1 = ant1-1
    ant2 = ant2-1    
    #Matrice de passage MB -> SB <=> 
    #(hh1,hh2,..,hhNa,hv1,hv2,...,hvNa,vv1,vv2,...,vvNa)->(hhi,hhj,hvi,hvj,vvi,vvj)
    P = np.zeros((Na_SB*3,Na*3))
    for k in range(3):
        P[Na_SB*k,k*Na+ant1]=1
        P[Na_SB*k+1,k*Na+ant2]=1
        
    Teb_SB= P.dot(MPMB.dot(P.T.conj()))        
    return Teb_SB
    
def criterion(R1,R2):
    """criterion used by Telbadini in the Two SM case """
        
    Crit=1-np.trace(R1.dot(R2))/(npl.norm(R1,'fro')*npl.norm(R2,'fro'))
    Crit = float(np.real(Crit)) #Crit est reel mais le type est complex
    return Crit

def monte_carlo_sm_separation(param,varia_taille_echant,N_real):
   """ Test de la méthode de séparation de Tebaldini par MonteCarlo. 
   Comptabilise l'echec moyen de la séparation pour 
   une taille d'echantillon donné.
   
   Pour un jeu de paramètre RVoG donnés par param, 
   la méthode est appliquée pour des tailles d'échantillons 
   contenues dans la liste vrai_taille_echant; ceci pour
   N_real réalisations
   
   **Entrée** : 
       * *param* : classe de paramètre RVoG
       * *varia_taille_echant* : liste contenant les tailles d'échantillon à tester
       * *N_real* : nombre de réalisations
       
   **Sortie**
       * *prop_echec* : taux d'echec.
   """ 
   
   #comptabilise le nombre d'echec moyen de la séparation pour 
   #une taille d'echantillon donné
   prop_echec = [0]*len(varia_taille_echant)
   set_fail_found = 0
   set_good_found = 0
   set_fail = ()
   set_good = ()
   for i_ech,taille_echant in enumerate(varia_taille_echant):
       set_fail_found = 0
       set_good_found = 0
       param.N = taille_echant #Ici on genere des données de la taille de la fenetre
       print '///////////////////////////////////////'
       print 'taille_echant',taille_echant
       for i_real in range(N_real):    
           if i_real % 25 == 0:
               print '==>realisation',i_real
           taille_test = taille_echant
           Np=3 #Nombre de polarisations
           Na=2 #Nombre d'antennes (recepteur)
           #param.display() affichage
           
           """cas bruité"""           
           tropi = TomoSARDataSet_synth(Na,param)                     
           mat_covariance_norm = tropi.get_covar_rect(taille_test)
           mat_covariance_norm = mat_covariance_norm/mat_covariance_norm[0,0]
           
           W_k=tropi.get_W_k_norm_rect(taille_test)
           W_k=W_k/W_k[0,0]
         
           """cas sans bruit"""
           """
           Ups = param.get_upsilon()
           mat_covariance_norm = Ups/Ups[0,0]
           W_k = UPS_to_MPMB(Ups)
           """
           #sm_separation  
           R_t,C_t = sm_separation(W_k,Np,Na)             
           
           interv_a,interv_b,Cond,alpha = search_space_definition(R_t,C_t)  
           #interv_a,interv_b,Cond,alpha,approx = search_space_definition_rob(R_t,C_t)  
           
           if sum(Cond.ravel()) == 0:
               
               prop_echec[i_ech] = prop_echec[i_ech] + 1
               print 'Echec! Aucun interval trouvé','iteration ',i_real,' i_ech',i_ech
               """
               np.set_printoptions(precision=3)
               print 'alpha',alpha; print 'cond'
               print Cond
               
               print 'R_t[0]' ; print R_t[0]
               print 'R_t[1]';print R_t[1]
               print 'C_t[0]'; print C_t[0]
               print 'C_t[1]'; print C_t[1]
               """
               #raw_input()
           else:
               """
               print "Interval trouvé"
               np.set_printoptions(precision=3)
               print 'alpha',alpha; print 'cond'
               print Cond 
               print 'R_t[0]' ; print R_t[0]
               print 'R_t[1]';print R_t[1]
               print 'C_t[0]'; print C_t[0]
               print 'C_t[1]'; print C_t[1]           
               """                 
       prop_echec[i_ech] = prop_echec[i_ech]*100/float(N_real)            
   return prop_echec
 

def monte_carlo_sm_separation_debug(param,varia_taille_echant,N_real):
   #FONCTION DE TEST DE DEUX SITUATIONS : 
   # 1 : good : intervalle de positivité trouvé
   # 2 : bad : intervalle de positivité non trouvé
   """ Test de la méthode de séparation de Tebaldini
   pour un jeu de paramètre RVoG donnés par param. 
   La méthode est appliquée pour des tailles d'échantillons 
   contenues dans la liste vrai_taille_echant; ceci pour
   N_real réalisations
   
   param: classe de paramètre RVoG
   varia_taille_echant: liste contenant les tailles d'échantillon à tester
   N_real: nombre de réalisations
   """ 
   
   #comptabilise le nombre d'echec moyen de la séparation pour 
   #une taille d'echantillon donné
   prop_echec = [0]*len(varia_taille_echant)
   set_fail_found = 0
   set_good_found = 0
   set_fail = ()
   set_good = ()
   for i_ech,taille_echant in enumerate(varia_taille_echant):
       set_fail_found = 0
       set_good_found = 0
       param.N = taille_echant #Ici on genere des données de la taille de la fenetre
       print '///////////////////////////////////////'
       print 'taille_echant',taille_echant
       for i_real in range(N_real):    
           if i_real % 1 == 0:
               print '==>realisation',i_real
           taille_test = taille_echant
           Np=3 #Nombre de polarisations
           Na=2 #Nombre d'antennes (recepteur)
           #param.display() affichage
           
           """cas bruité"""           
           tropi = TomoSARDataSet_synth(Na,param)                     
           mat_covariance_norm = tropi.get_covar_rect(taille_test)
           mat_covariance_norm = mat_covariance_norm/mat_covariance_norm[0,0]
           
           W_k=tropi.get_W_k_norm_rect(taille_test)
           W_k=W_k/W_k[0,0]
         
           """cas sans bruit"""
           """
           Ups = param.get_upsilon()
           mat_covariance_norm = Ups/Ups[0,0]
           W_k = UPS_to_MPMB(Ups)
           
           
           """
           #sm_separation  
           R_t,C_t = sm_separation(W_k,Np,Na)             
           #interv_a,interv_b,Cond,alpha = search_space_definition(R_t,C_t)  
           interv_a,interv_b,Cond,alpha,approx = search_space_definition_rob(R_t,C_t)  
           
           if sum(Cond.ravel()) == 0:
               
               prop_echec[i_ech] = prop_echec[i_ech] + 1
               print "Echec! Aucun interval trouvé"
               np.set_printoptions(precision=3)
               print 'alpha',alpha; print 'cond'
               print Cond
               """
               print 'R_t[0]' ; print R_t[0]
               print 'R_t[1]';print R_t[1]
               print 'C_t[0]'; print C_t[0]
               print 'C_t[1]'; print C_t[1]
               """
               #raw_input()
           else:
               print "Interval trouvé"
               np.set_printoptions(precision=3)
               print 'alpha',alpha; print 'cond'
               print Cond 
               print 'R_t[0]' ; print R_t[0]
               print 'R_t[1]';print R_t[1]
               print 'C_t[0]'; print C_t[0]
               print 'C_t[1]'; print C_t[1]           

                   
               if approx == 1:
                   set_fail = (R_t,C_t,mat_covariance_norm,interv_a,interv_b,alpha)
                   set_fail_found = 1
               else:
                   set_good = (R_t,C_t,mat_covariance_norm,interv_a,interv_b,alpha)
                   set_good_found = 1
                   
               print set_good_found,set_fail_found
 
               #raw_input()               
               
           if set_good_found == 1 and set_fail_found == 1:
               #raw_input()
               print 'Les 2 situations se sont produites'
               return prop_echec,set_good,set_fail
           """
           print '<===== Deboguage ======>'           
           np.set_printoptions(precision=3)
           print 'alpha',alpha; print 'cond'
           print Cond ; 
           print 'R_t[0]' ; print R_t[0]
           print 'R_t[1]';print R_t[1]
           print 'C_t[0]'; print C_t[0]
           print 'C_t[1]'; print C_t[1]
           print 'W_K'
           print W_k
           plt.figure()
           plt.axes(polar=True)
           plt.hold(True)
           polinsar_plot_cu(mat_covariance_norm)
           plt.plot(np.angle(R_t[0][0,1]),np.abs(R_t[0][0,1]),'*g')           
           plt.plot(np.angle(R_t[1][0,1]),np.abs(R_t[1][0,1]),'*b')
           raw_input()
           """
           

       prop_echec[i_ech] = prop_echec[i_ech]*100/float(N_real)            
   return prop_echec,set_good,set_fail
   
def ecart_angle_inclin_A_E(A_test,E_test,param,varia_taille_echant,\
                            N_real,sauv_data=1,\
                            plot_hist=0,sauv_hist=0,plot_coher=0):    
    
    """ Calcul du biais et variance d'estimation des angles des droites de 
    cohérences pour differents A et E.
    
    **Entrées** 
        * *A_test* : vec. des valeur de A 
        * *E_test* : vec. des valeur de E 
        * *param* : classe de paramètre RVoG
        * *varia_taille_echant* : liste contenant les tailles d'échantillon à tester
        * *N_real* : nombre de réalisations
        * *sauv_data* : booleen. =1 si on veut sauver les anlayses MonteCarl
        * *plt_hist* : booleen. =1 plot un histogramme
        * *plt_coher* : booleen. =1 plot la reg. de cohérence"""    

    for A in A_test:
        for E in E_test:
            param=mb.rvog_reduction(param,A,E)            
            Ups = param.get_upsilon()
            omega_vrai = polinsar_compute_omega12blanchi(Ups)
            theta_vrai,_= polinsar_estime_droite(omega_vrai)                
            Ha = 2*np.pi/param.k_z[0]
            varia_phig = [param.k_z[0]*param.z_g]         
            """
            polinsar_plot_cu(Ups)
            plt.show()
            print param.display()
            raw_input()
            """
            print '================= A={0} E={1} ================='.format(A,E)
            theta_pascale,theta_tebald,\
            theta_pascale_moy,theta_pascale_var,\
            theta_tebald_moy,theta_tebald_var,\
            theta_pascale_circ_moy,theta_pascale_circ_var,\
            theta_tebald_circ_moy,theta_tebald_circ_var=\
            ecart_angle_inclin_droite(param,varia_phig,varia_taille_echant,\
                                                            N_real,plot_coher)
            """            
            name_taille_ech = 'Taille echantillon {0} to {1} \t {2} elements'.format(min_taille,max_taille,nb_taille)
            name_param = 'phig '+str(varia_phig).strip('[]')+'\nA {1}\nE {2}'.format(varia_phig,A,E)
            """
            home_dir ='/home/capdessus/Python/Code_Pierre/'
            folder_name = 'data/angle_inclinaison_droite/'+bl.get_date_DM()
            sub_folder_name ='A_{0}_E_{1}/'.format(A,E)
            total_path = home_dir+folder_name+sub_folder_name
            if sauv_data:
                if os.path.isdir(total_path)==False:
                    os.makedirs(total_path)
                                   
                save_txt_config(param,varia_taille_echant,N_real,total_path)     
                np.save(total_path+'taille_echant',varia_taille_echant)
                np.save(total_path+'Ups',Ups)                    
                np.save(total_path+'theta_pascale_vrai',theta_vrai)
                np.save(total_path+'theta_pascale',theta_pascale)
                np.save(total_path+'theta_tebald',theta_tebald)                                    
                np.save(total_path+'theta_pascale_moy',theta_pascale_moy)
                np.save(total_path+'theta_pascale_var',theta_pascale_var)
                np.save(total_path+'theta_tebald_moy',theta_tebald_moy)
                np.save(total_path+'theta_tebald_var',theta_tebald_var)                              
            if(plot_hist):                
                fig_pascale = plt.figure()
                #histogramme des thetas de la premiere taille d'echant
                plt.hist(theta_pascale[0,0,:]*180/np.pi,bins=50)                                
                fig_tebald = plt.figure()
                plt.hist(theta_tebald[0,0,:]*180/np.pi,bins=50)                                
            if sauv_hist:                
                if os.path.isdir(total_path)==False:
                    os.makedirs(total_path)
                else:                       
                    plt.figure(fig_pascale.number)                    
                    plt.savefig(total_path+'histo_pascale.png',format='png')
                    plt.figure(fig_tebald.number)
                    plt.savefig(total_path+'histo_tebald.png',format='png')
                    
def ecart_angle_inclin_droite(param,varia_phig,varia_taille_echant\
                                            ,N_real,plot_cohe=0):
    """Observation de la difference d'angle entre les droites de cohérences
    obtenues par regression du nuage de coherence (Pascale) et celle obtenue
    par l'approche de Tebaldini,

    **Entrées** 
        * *param* : classe de paramètre RVoG
        * *varia_phig* : liste contenant les phases du sol a tester.
        * *varia_taille_echant* : liste contenant les tailles d'échantillon à tester
        * *N_real* : nombre de réalisations                
        * *plt_cohe* : booleen. =1 plot la reg. de cohérence"""    
        
    Na = 2
    Np = 3

    N_pts = 15
    gamma_a = np.zeros(N_pts,dtype='complex')
    gamma_b = np.zeros(N_pts,dtype='complex')
    taille = 20 #nbre de gamma des deux sm 

    theta_pascale_moy = np.zeros((len(varia_phig),len(varia_taille_echant)))
    theta_tebald_moy = np.zeros((len(varia_phig),len(varia_taille_echant)))
    theta_pascale_var = np.zeros((len(varia_phig),len(varia_taille_echant)))
    theta_tebald_var = np.zeros((len(varia_phig),len(varia_taille_echant)))
        
    theta_pascale_circ_moy = np.zeros((len(varia_phig),len(varia_taille_echant)))
    theta_tebald_circ_moy = np.zeros((len(varia_phig),len(varia_taille_echant)))
    theta_pascale_circ_var = np.zeros((len(varia_phig),len(varia_taille_echant)))
    theta_tebald_circ_var = np.zeros((len(varia_phig),len(varia_taille_echant)))
       
    theta_pascale = np.zeros((len(varia_phig),len(varia_taille_echant),N_real))
    theta_tebald = np.zeros((len(varia_phig),len(varia_taille_echant),N_real))
    
    
    Ups = param.get_upsilon()
    omega_vrai_blanc = polinsar_compute_omega12blanchi(Ups)
    theta_vrai,_ = polinsar_estime_droite(omega_vrai_blanc)

    for i_phi,phig in enumerate(varia_phig):    
        param.z_g = phig / param.k_z[0]
        for i_ech,taille_echant in enumerate(varia_taille_echant):
            #Ici on genere des données de la taille de la fenetre            
            param.N = taille_echant 
            print '///////////////////////////////////////'
            print 'phig',phig,'taille_echant',taille_echant
            for i_real in range(N_real):    
               if i_real % 500 == 0:
                   print '==>realisation',i_real
               
               #param.display() affichage
               tropi = TomoSARDataSet_synth(Na,param)
               W_k = tropi.get_W_k_rect(param,taille_echant,Na)
               W_k_norm = normalize_MPMB_PS_Tebald(W_k,Na)
               
               mat_covariance = MPMB_to_UPS(W_k,Na)        
               
               #W_k=W_k/W_k[0,0]
               #sm_separation  
               R_t,C_t,_ = sm_separation(W_k_norm,Np,Na)  
               
               interv_a,interv_b,Cond,alpha =\
                   search_space_definition(R_t,C_t,Na)  
                              
               if sum(Cond.ravel()) == 0:
                   np.set_printoptions(precision=3)
                   print 'Aucun interval trouvé'         
                   print 'alpha',alpha; print 'cond'
                   print Cond ; print 'R_t' ; print R_t
                   print 'C_t'; print C_t
                   return theta_pascale_moy,theta_pascale_var,\
                          theta_tebald_moy,theta_tebald_var,\
                          theta_pascale_circ_moy,theta_pascale_circ_var,\
                          theta_tebald_circ_moy,theta_tebald_circ_var

               mat_covariance_norm = normalize_T1T2(mat_covariance)
               omega = polinsar_compute_omega12blanchi(mat_covariance_norm)
               theta1,_ = polinsar_estime_droite(omega)
               theta_pascale[i_phi,i_ech,i_real] = theta1
               
               #formation des gamma_a et gamma_b                
               for i_a,a in enumerate(np.linspace(interv_a[0][0],interv_a[0][1],N_pts)):
                   gamma_a[i_a]=a*R_t[0][0,1]+(1-a)*R_t[1][0,1]
               for i_b,b in enumerate(np.linspace(interv_b[0][0],interv_b[0][1],N_pts)):
                   gamma_b[i_b]=b*R_t[0][0,1]+(1-b)*R_t[1][0,1]
                         
               gamma = np.hstack((gamma_a,gamma_b))
               theta_tebald[i_phi,i_ech,i_real] = bl.estime_line_svd(gamma,theta_vrai)
               if(plot_cohe):                   
                   pt.plot_cu_sm_possible(mat_covariance_norm,\
                                       R_t,interv_a,interv_b)
                   param.display()
                   print 'theta_pascale',theta1*180/np.pi
                   print 'theta_tebald',theta_tebald[i_phi,i_ech,i_real]*180/np.pi                   
                   print 'theta_vrai',theta_vrai*180/np.pi
                   plt.show()
                   raw_input()
                   plt.close('all')
                                       
            theta_pascale_moy[i_phi,i_ech] = np.mean(theta_pascale[i_phi,i_ech,:])
            theta_tebald_moy[i_phi,i_ech] = np.mean(theta_tebald[i_phi,i_ech,:])
            theta_pascale_var[i_phi,i_ech] = np.var(theta_pascale[i_phi,i_ech,:])
            theta_tebald_var[i_phi,i_ech] = np.var(theta_tebald[i_phi,i_ech,:])

    return theta_pascale,theta_tebald,\
           theta_pascale_moy,theta_pascale_var,\
           theta_tebald_moy,theta_tebald_var, \
           theta_pascale_circ_moy,theta_pascale_circ_var,\
           theta_tebald_circ_moy,theta_tebald_circ_var


def estime_angle_inclin(R_t,theta_vrai):
    
    #a=1 b=0
    R1 = 1*R_t[0]
    R2 = 1*R_t[1]
    gamma = [R1[0,1],R2[0,1]]
    bl.estime_line_svd(gamma,theta_vrai)

def denormalisation_teb(W_k,Na,C_g,C_v):
    """Renvoie les matrices Tground et Tvol 
    du modèle RVoG à partir de la matrice 
    de covar (base MPMB) et des réponses
    polarimétriques du sol et du volume.
    
    **Entrées**:
    * *W_k* : mat de covar (non {normalisé/blanchie})
    * *Na* : nombre d'antennes
    * *C_g*,*C_v* : réponses polar du sol et volume (SKP)
    
    **Sortie**:
    * *T_g* = *aTground*
    * *T_v* =*I1*Tvol*
    (Tvol,Tground) : rep. polar du sol et du volume 
    !!! Attention T_g n'est pas T_ground du RVog (T_v respectivement)
    """
    
    E = np.diag(np.diag(W_k.copy())) #Matrice des coeffs diagonaux de E    
    F = power(np.diag(np.array([E[0,0],E[Na,Na],E[2*Na,2*Na]])),0.5)    
    T_g = F.dot(C_g.dot(F))
    T_v = F.dot(C_v.dot(F))
    return T_g,T_v
    
def proximity_C_T(W_k,Na,R_t,C_t,vec_a,vec_b,param):
    """Calcule la distance entre les matrices
    C_k obtenues par décomposition skp (fonction sm_separation)
    avec les matrices T_vol et T_ground de départ
    
    **Entrées**
        * *W_k* : matrice de cohérence (base MPMB)
        * *Na* : nombre d\'antennes
        * *R_t* : liste des matrice de structures
        * *vec_a* : valeurs de a (associé a la cohér. du sol)
        * *vec_b* : valeurs de b (associé a la cohér. du volume)
        * *param* : object de type param_rvog
        
    **Sorties**:
        * *dis_g* : distance(Cg,Tground)
        * *dist_v* : distance(Cv,Tvol) """
            
    T_vol = param.T_vol
    T_ground  = param.T_ground    
    dist_g = np.zeros((vec_a.size,vec_b.size),dtype='float')
    dist_v = np.zeros((vec_a.size,vec_b.size),dtype='float')
    for idxa,a_test in enumerate(vec_a):
        for idxb,b_test in enumerate(vec_b):
            _,_,_,C_g_test,C_v_test =\
                        value_R_C(R_t,C_t,a_test,b_test)
            T_g,T_v = denormalisation_teb(W_k,Na,C_g_test,C_v_test)
            diff_g =T_g/T_g[0,0]-T_ground/T_ground[0,0]
            diff_v = T_v/T_v[0,0]-T_vol/T_vol[0,0]
            
            dist_g[idxa,idxb] = np.real(np.sqrt(np.trace(diff_g.\
                        dot(diff_g.T.conj()))))
            dist_v[idxa,idxb] = np.real(np.sqrt(np.trace(diff_v.\
                        dot(diff_v.T.conj())))) 
    return dist_g,dist_v
    
    
def save_txt_config_and_simu(param,varia_taille_echant,N_real,folder_path,namefile='config'):
    """Sauvegarde dans un fichier texte les paramètres de la classe param et les paramètres de simulations.
    
   **Entrées** 
        * *param* : classe de paramètre RVoG       
        * *varia_taille_echant* : liste contenant les tailles d'échantillon à tester
        * *N_real* : nombre de réalisations                
        * *folder_path* : nom de chemin du du dossier
        * *namefile* : nom du fichier"""

    if os.path.exists(folder_path) == False:
        #creation du dossier s'il n'existe pas
        os.makedirs(folder_path)

    min_taille = np.min(varia_taille_echant)
    max_taille = np.max(varia_taille_echant)
    nb_taille = np.size(varia_taille_echant)
    
    
    name_param ='N {0} k_z {1} theta {2}'.format(str(param.N),str(param.k_z),str(param.theta))\
    +' h_v {0} z_g {1} sigma_v {2}'.format(str(param.h_v),str(param.z_g),str(param.sigma_v))\
    +' phig {0}'.format(str(param.k_z[0]*param.z_g))\
    +'\nT_vol\n'\
    +'\n'.join(' '.join(str(cell) for cell in row) for row in param.T_vol)\
    +'\nT_ground\n'\
    +'\n'.join(' '.join(str(cell) for cell in row) for row in param.T_ground)    
    
    name_taille_ech = '\nTaille echantillon {0} to {1} \t {2} elements'.format(min_taille,max_taille,nb_taille)
    name_N_real='\nNreal {0}'.format(N_real)
    A,E,_,_,_=param.get_invariant()
    name_param_rvog = '\nA {0}\nE {1}'.format(A,E)
    txt = name_param+name_param_rvog+name_taille_ech+name_N_real
    
    
    file_log = open(folder_path+namefile,'w')
    
    file_log.write(txt)
    file_log.close()
 
    

