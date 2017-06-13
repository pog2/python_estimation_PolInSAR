# -*- coding: utf-8 -*-
""" Classe PolInSARDataSet et routine associée"""
import sys
import numpy as np
import numpy.linalg as npl
import matplotlib.mlab as mat
import matplotlib.pyplot as plt
import pymela
import zone_analyse as za
import stat_SAR as st
import load_param as lp
#import pylab
plt.ion()

class TomoSARDataSet(object):
    def __init__(self,Na, listImageName,listImageNameHamb,sigma=0.4):
        self.imgMhh = [None]*Na
        self.imgMhv = [None]*Na
        self.imgMvh = [None]*Na
        self.imgMvv = [None]*Na
        self.imgHa = [None]*(Na-1)       
        imageNameMhh = [None]*Na                
        imageNameMhv = [None]*Na                
        imageNameMvh = [None]*Na                
        imageNameMvv = [None]*Na                
        imageNameHa = [None]*Na
        #Chargement des noms des fichiers de donnés
        for j,Name in enumerate(listImageName):
            imageNameMhh[j] = str(Name)
            imageNameMhv[j] = imageNameMhh[j].replace('Hh','Hv')            
            imageNameMvh[j] = imageNameMhh[j].replace('Hh','Vh')
            imageNameMvv[j] = imageNameMhh[j].replace('Hh','Vv')
        #chargement des noms des fichiers de hauteurs d'ambiguité
        #et chargment des fichiers de hauteurs
        for j,Name in enumerate(listImageNameHamb):   
            imageNameHa[j] = Name
            self.imgHa[j] = pymela.Image(imageNameHa[j],load=True)   
        #chargement des images
        for j in range(Na):
            self.imgMhh[j] = pymela.Image(imageNameMhh[j], load=True)
            self.imgMhv[j] = pymela.Image(imageNameMhv[j], load=True)
            self.imgMvh[j] = pymela.Image(imageNameMvh[j], load=True)
            self.imgMvv[j] = pymela.Image(imageNameMvv[j], load=True)             
              
        self.Na = Na #Nombre d'acquisitions(antennes)
        self.nrange = self.imgMhh[0].largeur
        self.nazi = min(self.imgMhh[0].longueur, self.imgMhh[0].longueur)     
        self.extinction = np.log(10.**(sigma/20.))
       
    def get_nrange(self):
        return self.nrange
    def get_nazi(self):
        return self.nazi
    def get_near_range(self):    
        return self.imgMhh[1]["PREMIERE PORTE DISTANCE"]
    def get_pas_distance(self):    
        return self.imgMhh[1]["PAS DISTANCE"]
    def get_pas_azimut(self):    
        return self.imgMhh[1]["PAS AZIMUT"]
    def get_hauteur_sol(self):
        return self.imgMhh[1]["HAUTEUR SOL ORIGINE"]

    def compute_incidence(self, irange):
        range = self.get_near_range() + irange*self.get_pas_distance()
        return np.arccos(self.get_hauteur_sol()/range)
 
    def get_data_teb_rect(self, il0, il1, ic0, ic1):
        """Récupère un tableau de dimension (Nax3) x N contenant les valeurs des 
        pixels associes à la zone azimut (il0,il1) distance (ic0, ic1)
        Attention la il1 ligne et ic1 colonne ne sont pas prises en compte
        
        Les données sont rangées selon la méthode de Tebaldini (MPMB):
        y = (hh1,hh2,..,hhNa,hv1,hv2,..,hvNa,vv1,vv2,..,vvNa)
            
        **Entrées**
            * *il0* , *il1* : indices azimut (ligne)
            * *ic0* , *ic1* : indices distance (colonne)
            
        **Sortie**
            * *data* : *array*. Données extraites."""
            
        taille = (il1-il0)*(ic1-ic0)
        data = np.zeros((3*Na,taille), 'complex64')
        for j in range(Na):
            data[j,:] = self.imgMhh[j][il0:il1,ic0:ic1].ravel()
            data[j+Na,:] = (self.imgMhv[j][il0:il1,ic0:ic1].ravel() + self.imgMvh[j][il0:il1,ic0:ic1].ravel()) /np.sqrt(2)
            data[j+2*Na,:] = self.imgMvv[j][il0:il1,ic0:ic1].ravel()
        return data        
        
    def get_data_rect(self, il0, il1, ic0, ic1):
        """Récupère un tableau de dimension (3xNa) x N contenant les valeurs des 
        pixels associes à la zone azimut (il0,il1) distance (ic0, ic1)
        Attention la il1 ligne et ic1 colonne ne sont pas prises en comtpe
            
        Les données sont rangées de manière classiques:
        y = (hh1,hv1,vv1,hh2,hv2,vv2,...,hhNa,hvNa,vvNa)
        
        **Entrées**
            * *il0* , *il1* : indices azimut (ligne)
            * *ic0* , *ic1* : indices distance (colonne)
            
        **Sortie**
            * *data* : *array*. Données extraites."""
            
        taille = (il1-il0)*(ic1-ic0)
        data = np.zeros((3*Na,taille), 'complex64')
        
        for j in range(Na):
            data[3*j,:] = self.imgMhh[j][il0:il1,ic0:ic1].ravel()
            data[3*j+1,:] = (self.imgMhv[j][il0:il1,ic0:ic1].ravel() + self.imgMvh[j][il0:il1,ic0:ic1].ravel()) /np.sqrt(2)
            data[3*j+2,:] = self.imgMvv[j][il0:il1,ic0:ic1].ravel()            
        return data
      
    def get_data_synth(self,nb_echant):
        """ Génère un tableau *3Naxnb_echant* de données PolInSAR 
        multibaseline synthétique.
        
        **Entrée**
            * *nb_echant* : taille de l'echantillon
        
        **Sortie**
            * *data* : *array*. Données PolinSAR synthétique."""
            
        Ups = st.generate_UPS(param)
        nc = 3*Na #nombre de composantes du vecteur d'observation. 
        k = st.generate_PolInSAR(Ups,nb_echant)               
        data = np.zeros((3*Na,taille), 'complex64')    
        for j in range(Na):
            data[3*j,:] = k[j*3,:]
            data[3*j+1,:] = k[j*3+1,:]
            data[3*j+2,:] =  k[j*3+2,:]
        return data

    def get_ha_all_rect(self, il0, il1, ic0, ic1):
        """Retourne la liste des hauteurs d'ambiguité au centre de la fenêtre 
        il0:il1 x ic0:ic1.
        
        La liste est constituée des Ha correspondant aux baselines
        de la forme B1,i (i indice de l'antenne): B12,B13,B14,...
        
        **Entrées**
            * *il0* , *il1* : indices azimut (ligne)
            * *ic0* , *ic1* : indices distance (colonne)
            
        **Sortie**
            * *data* : *array*. Données extraites"""
        
        il = (il0+il1)//2.0
        ic = (ic0+ic1)//2.0   
        H=[self.imgHa[i][il,ic] for i in range(self.Na)]    
        return H

    def get_ha_single_rect(self, il0, il1, ic0, ic1,ant1,ant2):
        """Retourne la hauteur d'ambiguité entre *ant1* et *ant2*
        au centre de la fenêtre  il0:il1 x ic0:ic1.
                
        **Entrées**
            * *il0, il1* : indices azimut (ligne)
            * *ic0, ic1* : indices distance (colonne)
            * *ant1, ant2* : indices des antennes
            
        **Sortie**
            * *ha_single* : *scalar*. Hauteur d'ambiguité"""

        H=self.get_ha_all_rect(self, il0, il1, ic0, ic1)
        
        if ant1 == ant2:          
            print "Attention! : Même valeur! ant1=",ant1,"ant2=",ant2
            raise RuntimeError('ant1 doit être différent de ant2')
        else:
            ha_single= 1/((-1/H[i]+1/H[j]))
        return ha_single
        
    def get_covar_rect(self, il0, il1, ic0, ic1):
        """Renvoie la matrice de covariance estimée sur la fenêtre 
        il0:il1 x ic0:ic1.
        
        Les données sont rangées de manière classiques:
        y = (hh1,hv1,vv1,hh2,hv2,vv2,...,hhNa,hvNa,vvNa)
        
        **Entrées**
            * *il0* , *il1* : indices azimut (ligne)
            * *ic0* , *ic1* : indices distance (colonne)
            
        **Sortie**
            * *Ups* : *array*. Matrice de covariance. """

        data = self.get_data_rect(il0, il1, ic0, ic1)
        Ups = data.dot(data.T.conj())
        return Ups
        
    def get_W_rect(self,il0, il1, ic0, ic1):
        """Renvoie la matrice de covariance en coordonnées MPMB estimée sur 
        la fenêtre il0:il1 x ic0:ic1.
        
        **Entrées**
                * *il0* , *il1* : indices azimut (ligne)
                * *ic0* , *ic1* : indices distance (colonne)
            
        **Sortie**
            * *W* : *array*. Matrice de covaricance (coordonnées MPMPB)"""
            
        taille = (il1-il0)*(ic1-ic0)
        data = self.get_data_teb_rect(il0, il1, ic0, ic1)
        W = data.dot(data.T.conj())/taille
        return W
        
    def get_W_norm_rect(self, il0, il1, ic0, ic1,type_norm='ps+tebald'):        
        """Renvoie la matrice de covariance normalisée en coordonnées MPMB.
        
        **Entrées**
            * *param*: classe de paramètre rvog
            * *nb_echant*: taille d'echantillon            
            * *type_norm*: type de normalisation 
            
                * *mat+ps*: Application d'une normalisation+egalisation des rep polarmietrique
                * *mat+ps+tebald*: precedent + normalisation selon chque recepteur et polar
                * *ps+tebald*: egalisation des reps polar+tebald
                
        **Sortie**
            * *covar* : matrice de covariance"""
        
        W = self.get_W_rect(il0, il1, ic0, ic1)
        
        if type_norm =='mat+ps':
            W_norm = normalize_MPMB_mat_PS(W,self.Na)
        elif type_norm =='ps+tebald':
            W_norm,_ = normalize_MPMB_PS_Tebald(W,self.Na)
        elif type_norm == 'mat+ps+tebald':
            W_norm = normalize_MPMB_mat_PS_Tebald(W,self.Na)
        else:
            print 'Attention Type de normalisation inconnu ! '
        return W_norm
        
        
    def polinsar_computeh_from_psi(self, pos_center, psi):
        """j: indice de Ha        
        pos_center est un tuple ligne, colonne qui indique le centre de la région, 
        psi est l'angle d'ouverture de la droite de cohérence"""
        
        j=0
        u_droite = np.cos(psi)-1 +1j*(np.sin(psi))
        costeta = np.cos(self.compute_incidence(pos_center[1]))
        ha = self.imgHa[j][pos_center[0],pos_center[1]]
        kz = -2*np.pi/ha
        #print 'kz ',kz,'  ha',ha
        extinction = self.extinction
        hmax = min(50.,abs(ha))
        hmin = 5.
        cmin = (u_droite* np.conj(polinsar_gamav(costeta,kz,extinction,hmin) - 1.)).imag
        cmax = (u_droite* np.conj(polinsar_gamav(costeta,kz,extinction,hmax) - 1.)).imag
        if cmin*cmax > 0:
            return hmax
        while (hmax-hmin) > 0.05:
            c = (u_droite*np.conj(polinsar_gamav(costeta,kz,extinction,(hmin+hmax)/2.)-1.)).imag
            if (c*cmin > 0):
                hmin = (hmin+hmax)/2
                cmin = c
            else:
                hmax = (hmin+hmax)/2
                cmax = c
        return (hmax+hmin)/2       
        
    def polinsar_computesigma_from_psi_h(self,pos_center, psi, h):
        """j: indice de baseliene"""   
        """pos_center est un tuple ligne, colonne qui indique le centre de la région, 
        psi est l'angle d'ouverture de la droite de cohérence"""
        
        u_droite = np.cos(psi)-1 +1j*(np.sin(psi))
        costeta = np.cos(self.compute_incidence(pos_center[1]))
        ha = self.imgHa[pos_center[0],pos_center[1]]
        kz = -2*np.pi/ha
        #print 'kz ',kz,'  ha',ha
        temp=np.linspace(0.1,0.6,51)
        extinction =  np.log(10.**(temp/20.))
        # droite est ax+by+c=0 avec a=-sin(psi) b=cos(psi) - 1 et c=sin(psi)        
        a=-np.sin(psi)
        b=np.cos(psi)-1.
        c=-a
        dist=np.zeros(51)
        for i in range(51):  
            extinct=extinction[i]
            g=polinsar_gamav(costeta,kz,extinct,h)
            y=np.imag(g)
            x=np.real(g)
            dist[i]=abs(a*x+b*y+c)
        minindex=dist.argmin()
        return np.log10(np.exp(extinction[minindex]))*20
                
    def polinsar_estime_tvol_tground(self,covar,pos_center):
        """ calcul le tvol et tground
        covar est la matrice de covariance
        pos_centerc sont la position ligne, colonne du centre de la zone"""
        
        covarn=normalize_T1T2(covar)
        costeta=np.cos(self.compute_incidence(pos_center[1]))
        critere='hh-hv'
        phig,psi = polinsar_calcul_phig_psi(covarn,critere)
        hv=self.polinsar_computeh_from_psi(pos_center,psi)
        kz = -2*np.pi/self.imgHa[0][pos_center[0],pos_center[1]] #!!![0] car 
        #en SB une seule hauteur d'ambiguité!!!! a généraliser !!! 
        alpha=2*self.extinction/costeta
        a=np.exp(-alpha*hv)
        I1=(1-a)/alpha
        I2=(np.exp(complex(0.,1.)*kz*hv)-a)/(complex(0,1)*kz+alpha)
        omega=covarn[0:3,3:]
        T1=covarn[0:3,0:3]
        ombar=np.exp(-complex(0.,1.)*phig)*omega
        ombar=(ombar+np.conj(ombar).T)/2.
        tvol= (ombar-T1)/(0.5*(I2+np.conj(I2))-I1)
        tground=(T1-I1*tvol)/a
        return tvol, tground, I1, a

    def rvogsb_estime_phig_hv_rect(self,il0,il1,ic0,ic1):
        covar=self.get_covar_rect(il0,il1,ic0,ic1)
        covarn=normalize_T1T2(covar) #on ajuste la calibration radiométrique de l'image esclave
        critere='hh-hv'
        pos_center=[(il1+il0)/2,(ic1+ic0)/2]
        #critere='hh-vv'
        #critere='hhmvv-hv'
        phig,psi = polinsar_calcul_phig_psi(covar,critere)
        hv=self.polinsar_computeh_from_psi(pos_center,psi)
        return phig,hv
        
    def rvogsb_bcr_rect(self,il0,il1,ic0,ic1):
        covar=self.get_covar_rect(il0,il1,ic0,ic1)
        covarn=normalize_T1T2(covar)
        pos_center=[(il1+il0)/2,(ic1+ic0)/2]
        costeta=np.cos(self.compute_incidence(pos_center[1]))
        critere='hh-hv'
        phig,psi = polinsar_calcul_phig_psi(covarn,critere)
        hv=self.polinsar_computeh_from_psi(pos_center,psi)
        kz = -2*np.pi/self.imgHa[pos_center[0],pos_center[1]]
        alpha=2*self.extinction/costeta
        a=np.exp(-alpha*hv)
        I1=(1-a)/alpha
        I2=(np.exp(complex(0.,1.)*kz*hv)-a)/(complex(0,1)*kz+alpha)
        omega=covarn[0:3,3:]
        T1=covarn[0:3,0:3]
        ombar=np.exp(-complex(0.,1.)*phig)*omega
        ombar=(ombar+np.conj(ombar).T)/2.
        tvol= (ombar-T1)/(0.5*(I2+np.conj(I2))-I1)
        tground=(T1-I1*tvol)/a
        b=np.exp(1j*phig)
        matrix_derive=calcul_matrix_derive(a,b,I1,I2,alpha,kz,hv,tvol,tground,omega)
        upsilon_i=covar_inverse(covarn)
        fisher=np.zeros((20,20),dtype='complex')
        for i in range(20):
            for j in range(20):
                fisher[i,j]=np.trace( np.dot( np.dot(upsilon_i,matrix_derive[:,:,i]),
                                          np.dot(upsilon_i,matrix_derive[:,:,j]) ) )
                                        
        fisher=np.dot(0.5,fisher+np.conj(np.transpose(fisher)))
        bcr =covar_inverse(fisher)
#        print 'Ecart type sur zg pour 100 pixels', np.sqrt(float(tropi35.rvogsb_bcr_rect(2402,2605+1,1312,1616+1)[0])/100)
#        print 'Ecart type sur hv pour 100 pixels', np.sqrt(float(tropi35.rvogsb_bcr_rect(2402,2605+1,1312,1616+1)[1])/100)

        return bcr[18,18],bcr[19,19]
        
    def rvogsb_image(self,il0,il1,ic0,ic1,dwidth_l=5,dwidth_c=5):
        """ Compute deux images phig et hv pour une zone définie
        par les 4 premiers indices avec une fenetre glissante de demi largeur width_l,width_c"""
        
        im_phig=np.zeros((il1-il0,ic1-ic0))
        im_hv=np.zeros((il1-il0,ic1-ic0))
        for il in range(il0+dwidth_l,il1-dwidth_l):
            if il % 100 == 0:
                print il,
            for ic in range(ic0+dwidth_c,ic1-dwidth_c):
                phig,hv=self.rvogsb_estime_phig_hv_rect(il-dwidth_l,il+dwidth_l+1,ic-dwidth_c,ic+dwidth_c+1)
                im_phig[il-il0,ic-ic0]=phig
                im_hv[il-il0,ic-ic0]=hv
        return im_phig,im_hv
        
    def rvogsb_attenuation_polar(self,il0,il1,ic0,ic1):
        covar=self.get_covar_rect(il0,il1,ic0,ic1)
        covarn=normalize_T1T2(covar) #on ajuste la calibration radiométrique de l'image esclave
        critere='hh-hv'
        pos_center=[(il1+il0)/2,(ic1+ic0)/2]
        #critere='hh-vv'
        #critere='hhmvv-hv'
        phig,psi = polinsar_calcul_phig_psi(covar,critere)
        hv=self.polinsar_computeh_from_psi(pos_center,psi)        
        ghh = covarn[0,3]/covarn[0,0]
        ghv = covarn[1,4]/covarn[1,1]
        gvv = covarn[2,5]/covarn[2,2]
        psihh=np.pi+2*(np.angle(ghh-np.exp(+phig*1j))-np.angle(np.exp(+phig*1j)))
        psihv=np.pi+2*(np.angle(ghv-np.exp(+phig*1j))-np.angle(np.exp(+phig*1j)))
        psivv=np.pi+2*(np.angle(gvv-np.exp(+phig*1j))-np.angle(np.exp(+phig*1j)))
        if (psihh < 0):
            psihh = psihh +2*np.pi
        if (psihv < 0):
            psihv = psihv +2*np.pi
        if (psivv < 0):
            psivv = psivv +2*np.pi
            
        print psi,psihh,psihv,psivv
        h = self.polinsar_computeh_from_psi( pos_center, psi)
        sigmahh=self.polinsar_computesigma_from_psi_h(pos_center,psihh,h)
        sigmahv=self.polinsar_computesigma_from_psi_h(pos_center,psihv,h)
        sigmavv=self.polinsar_computesigma_from_psi_h(pos_center,psivv,h)
       
        return psi,sigmahh,sigmahv,sigmavv
def normalize_MPMB_mat_PS(W,Na):         
    """Normalisation de la matrice de covariance pour imposer 
    la stationnarité polarimétrique (PS)
    (hh1=hh2=..=hhNa; hv1=hv2...=hvNa et vv1=vv2=...=vvNa)
    
    Version inspirée de la méthode de Pascale: \n
    ->Passage dans la base lexicographie \n
    ->Normalisation (operation matricielle)\n
    ->Passage base MPMB (operation matricielle)
    
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
        
def normalize_MPMB(y,Na):   
    fact_hh=np.zeros(Na-1)
    fact_vv=np.zeros(Na-1)
    fact_hv=np.zeros(Na-1)
    y_norm = np.zeros(y.shape,'complex64')
    for i in range(Na-1):
        fact_hh[i]=np.sqrt((np.real(y[0,:].dot(np.conj(y[0,:].T))))/(np.real(y[i+1,:].dot(np.conj(y[i+1,:].T)))))
        fact_vv[i]=np.sqrt((np.real(y[2*Na,:].dot(np.conj(y[2*Na,:].T))))/(np.real(y[2*Na+i+1,:].dot(np.conj(y[2*Na+i+1,:].T)))))
        fact_hv[i]=np.sqrt(fact_hh[i]*fact_vv[i])
 
        y_norm[0,:] = y[0,:]
        y_norm[Na,:] = y[Na,:]
        y_norm[2*Na,:] = y[2*Na,:]
    for i in range(Na-1):
        y_norm[i+1,:]=y[i+1,:]*fact_hh[i]
        y_norm[i+Na+1,:]=y[i+Na+1,:]*fact_hv[i]        
        y_norm[i+2*Na+1,:]=y[i+2*Na+1,:]*fact_vv[i]
    return y_norm   
        
def normalize_T1T2(covar):
    calfac=np.zeros(3,'float')
    covarn=np.zeros((6,6),'complex64')
    calfac[0]=np.sqrt(np.real(covar[0,0]/covar[3,3]))
    calfac[1]=np.sqrt(np.real(covar[1,1]/covar[4,4]))
    calfac[2]=np.sqrt(np.real(covar[2,2]/covar[5,5]))
    covarn[:]=covar[:]
    for i in range(3):
        for j in range(3):
            covarn[j,i+3]=covar[j,i+3]*calfac[i]
            covarn[j+3,i]=covar[j+3,i]*calfac[j]
            covarn[i+3,j+3]=covar[i+3,j+3]*calfac[i]*calfac[j]
    T11=covarn[0:3,0:3].copy()
    T22=covarn[3:,3:].copy()
    covarn[0:3,0:3]=(T11+T22)/2.
    covarn[3:,3:]=(T11+T22)/2.
    return covarn        
        
       
def normalize_T1T2_tomo(covar,Na):
    #Normalisation de la matrice de cov en coord MPMB
    #11/02 : ne fonctionne pas / difficile a mettrre en place (indices compliqués)
    calfrac=np.zeros((Na,3),'float')
    covarn=np.zeros((6,6),'complex64')
    
    for i in range(Na-1):
        #On a 3(Na-1) facteur de normalisations
        calfrac[i,0]=np.sqrt(np.real(covar[0,0]/covar[1+i,1+i]))
        calfrac[i,1]=np.sqrt(np.real(covar[Na,Na]/covar[Na+i+1,Na+i+1]))
        calfrac[i,2]=np.sqrt(np.real(covar[2*Na,2*Na]/covar[2*Na+i+1,2*Na+i+1]))
        
    """calfac[0]=np.sqrt(np.real(covar[0,0]/covar[3,3]))
    calfac[1]=np.sqrt(np.real(covar[1,1]/covar[4,4]))
    calfac[2]=np.sqrt(np.real(covar[2,2]/covar[5,5]))
    covarn[:]=covar[:]"""
    """            
    for i in range(3):
        for j in range(3):
            covarn[j,i+3]=covar[j,i+3]*calfac[i]
            covarn[j+3,i]=covar[j+3,i]*calfac[j]
            covarn[i+3,j+3]=covar[i+3,j+3]*calfac[i]*calfac[j]
    T11=covarn[0:3,0:3].copy()
    T22=covarn[3:,3:].copy()
    covarn[0:3,0:3]=(T11+T22)/2.
    covarn[3:,3:]=(T11+T22)/2.
    """  
    for i in range(Na):
        for j in range(3):
            #Pour Omega ...
            covarn[j*Na,i*Na+1]=covarn[j*Na,i*Na+1]*calfrac[i,0]
            covarn[j*Na,i*Na+1]=covarn[j*Na,i*Na+1]*calfrac[i,1]
            covarn[j*Na,i*Na+1]=covarn[j*Na,i*Na+1]*calfrac[i,2]
            #PourT2
            covarn[j*Na+1,i*Na+1]=covarn[j*Na+1,i*Na+1]*calfrac[i,0]*calfrac[j,0]
            covarn[j*Na+1,i*Na+1]=covarn[j*Na+1,i*Na+1]*calfrac[i,1]*calfrac[j,0]
            covarn[j*Na+1,i*Na+1]=covarn[j*Na+1,i*Na+1]*calfrac[i,2]*calfrac[j,0]
    #Remplissage des zeros (covarn est hermitienne)
    for i in range(3*Na):
        for j in range(3*Na):
            if covarn[i,j] == 0:
                covarn[i,j] = covarn[j,i] 
    return covarn
    
def sqrt_inverse(covar):
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
    """ Calcule la matrice omega blanchi au sens de FF
    Attention la matrice doit être une 6x6"""
    t11=covar[0:3,0:3]
    t22=covar[3:6,3:6]
    omega=covar[0:3,3:6]
    omega_blanchi=sqrt_inverse(t11).dot(omega.dot(sqrt_inverse(t22)))
    return omega_blanchi

def polinsar_compute_omega12blanchi(covar):
    """ Calcule la matrice omega blanchi au sens de FF
    ceal devrait marcher aussi pour la CP"""
    temp=np.vsplit(covar,2)
    bloc=[np.hsplit(temp[0],2),np.hsplit(temp[1],2)]
    omega_blanchi=sqrt_inverse(bloc[0][0]).dot(bloc[0][1].dot(sqrt_inverse(bloc[1][1])))
    return omega_blanchi
    
def polinsar_estime_droite(omega):
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
#    print 'test de polinsar_estime_droite',theta2,d2,theta1,d1,c1,c2
    return theta2,d2
    
def polinsar_ground_selection(covar,phi1,phi2,critere):
    """ Suivant le critère choisi, on sélectionne la phase du sol entre phi1 et phi2
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
    phi1=np.pi/2-theta2+np.arccos(d2)
    phi2=np.pi/2-theta2-np.arccos(d2)
    
    if phi1 < 0:
        phi1 += 2*np.pi
    if (phi2 < 0):
        phi2=phi2+2*np.pi
    phi1,phi2=polinsar_ground_selection(covar,phi1,phi2,critere)
    return phi1,phi2

def polinsar_calcul_phig_psi(covar,critere='hh-hv'):
    """ effectue le calcul de la phase du sol et de ouverture angulaire
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
    """ Plot the cohérence region associated with the 6x6 covariance matrix 
    covar"""
    covarn = normalize_T1T2(covar)
#    plt.figure(1)
    plt.axes(polar=True)
#    plt.title='test'
    T1=covarn[:3,:3]
    omega = covarn[:3,3:]
    # tracer plusieurs cohérences obtenues de manière alléatoire
    k=np.random.randn(500,3) + 1j*np.random.randn(500,3)
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
    omegab=polinsar_compute_omega12blanchi(covarn)
    plt.plot(phig,1.,'ko')
    plt.text(1.,1.2,title)
#    pylab.show()
    return    
def polinsar_plot_cu_orientation(covar,title=' CU'):
    """ Plot the cohérence region associated with the 6x6 covariance matrix 
    covar - explore the orientation effect"""
    covarn = normalize_T1T2(covar)
#    plt.figure(num)
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
    
#    pylab.show()
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

"""A ranger dans une biblio plus maths"""
def is_hermitian_num(A):   
    """Determine si A est hermitienne "numeriquement"
       i,e si ||A-A.H| < eps """
    eps=1e-3
    if(npl.norm(A-np.conj(A.T),'fro')<eps):
        return True
    else:
        return False
def sm_separation(W,Np,Na=2,Ki=2):
    
    """Separation de plusieurs mecanismes de diffusion à partir de la matrice 
    de covariance (base MPMB)
    
    Implémentation de la méthode de Telbaldini (méthode SKP).
    
    **Entrées** : 
        * *W* : matrice de covariance des données (base MPMB).
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
       
    return R_t,C_t
    
def ground_selection_MB(R_t,interv_a,interv_b):
    """Selection du sol selon le criètre : \|coherence\| la plus elevée 
    
    **Entrées** : 
        * *R_t* : liste de matrices contenant les matrices de structures
                  du sol et du volume
        * *interv_a* : intervale de valeur possible pour a (cohérence du sol)
        * *interv_b* : intervale de valeur possibles pour b (cohérence du volume)

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
    
def gamma_a_b(R_t,a,b):
    """Renvoie la coherence de chaque R_k pour une valeur du couple 
    (a,b) (Ki=2)"""
    
    gamma1 = a*R_t[0][0,1]+(1-a)*R_t[1][0,1]
    gamma2 = b*R_t[0][0,1]+(1-b)*R_t[1][0,1]
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
    """Renvoie les racines du polynomes verfié par \|gamma12\|² 
    dans le cas de 3 Baselines et 
    les hypothèses *g12* = g12g12xalphaxexp(ixbeta)x
    et *g23* = *g12* """ 
    
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
        * *interv_a* , *interv_b* : intervals possibles pour les val. de a et b
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
        * *interv_a* , *interv_b* : intervals possibles pour les val. de a et b
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
        * *interv_a* , *interv_b* : intervals possibles pour les val. de a et b
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
        
    **Sortie : *A* : matrice dont les colonnes sont issues du vec. a.
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
    """criterion used by Telbadini 
    in the Two SM case """
    Crit=1-np.trace(R1.dot(R2))/(npl.norm(R1,'fro')*npl.norm(R2,'fro'))
    Crit = np.real(Crit) #Crit est reel mais le type est complex
    return Crit

def thrash():
    
    #TEST ESTIMATION PAR METHODE RVoG
    step=1e-4
    sigma = 0.4
      
    critere='hh-hv'
    mat_covariance_norm_synth = tropi35.get_covar_rect(il0,il1,ic0,ic1)
    mat_covariance_norm_synth = normalize_T1T2(mat_covariance_norm_synth)
    
    phig,psi = polinsar_calcul_phig_psi(mat_covariance_norm_synth,critere)
    h_v_inters = tropi35.polinsar_computeh_from_psi(pos_center,psi)
    print h_v_inters

    
    a_test = np.arange(interv_a[0][0],interv_a[0][1],step)
    b_test =np.arange(interv_b[0][0],interv_b[0][1],step)
    h_v_inters = np.zeros((size(a_test),size(b_test)))
    
    for i,a in enumerate(a_test):#?? un seul interv possible pour a 
        for j,b in enumerate(b_test):
            #extraction des cohérences interféro 
            gamma_1 = a*R_t[0][0,1]+(1-a)*R_t[1][0,1]
            gamma_2 = b*R_t[0][0,1]+(1-b)*R_t[1][0,1]
            vec_gamma = [gamma_1,gamma_2]
            #determination de gamma_sol et gamma_veg
            gamma_sol = vec_gamma[np.argmax(np.abs(vec_gamma))]
            gamma_veg = vec_gamma[np.argmin(np.abs(vec_gamma))]
            
            Phi_g = np.angle(gamma_sol)
            h_v = 5;step_hv = 0.1
            critere='hh-hv'
            phig,psi = polinsar_calcul_phig_psi(mat_covariance_norm,critere)
            h_v_inters[i,j] = tropi35.polinsar_computeh_from_psi(pos_center,psi)
            
    #Determionation du couple (a,b) pour lequel h_v_inters[a,b] 
    # est le plus proche du vrai : h_v_rvog 
    #Calcul de h_v par RvoG
    covar=tropi35.get_covar_rect(il0,il1,ic0,ic1)
    covarn=normalize_T1T2(covar)
    costeta=np.cos(tropi35.compute_incidence(pos_center[1]))
    critere='hh-hv'
    phig,psi = polinsar_calcul_phig_psi(covarn,critere)
    h_v_rvog=tropi35.polinsar_computeh_from_psi(pos_center,psi)
     
    ###Condtion positivité de R1 R2 C1 et C2 par diagonlisation conjointe
    #TEST SUR PLEIN DE VALEUR DE a et b
    #a_test=np.arange(0.9,1.1,0.001) 
    #b_test= np.arange(0.9,1.1,0.001) 
    #Mais seuls les intervals interv_a et interv_b suffisent
    
    R_t_diag = [None]*Ki
    C_t_diag = [None]*Ki
    #diag conjointe
    R_t_diag[0],R_t_diag[1],T_R,_ = ejd(R_t[0],R_t[1])
    C_t_diag[0],C_t_diag[1],T_C,_ = ejd(C_t[0],C_t[1])
    step=0.001
    #pour a
    a_valable = []    
    a_test=np.arange(0.9,1.1,step) 
    gamma_valable_a = []
    for i,a in enumerate(a_test):
        gamma_a=a*R_t[0][0,1]+(1-a)*R_t[1][0,1]
        Cond_a_b = positivity_condition(R_t_diag,C_t_diag,a,-150)
        if Cond_a_b[0]==True:#on ne s'interesse qu'a la composante contenant a
            a_valable.append(a)
            gamma_valable_a.append(gamma_a)
    #pour b                
    b_valable = []
    b_test = np.arange(0.9,1.1,step) 
    gamma_valable_b = []
    for i,b in enumerate(b_test):
        gamma_b=b*R_t[0][0,1]+(1-b)*R_t[1][0,1]
        Cond_a_b = positivity_condition(R_t_diag,C_t_diag,150,b)
        if Cond_a_b[1]==True:
            b_valable.append(b)
            gamma_valable_b.append(gamma_b)

    ##Version avec test des interval
    
    R_t_diag = [None]*Ki
    C_t_diag = [None]*Ki
    #diag conjointe
    R_t_diag[0],R_t_diag[1],T_R,_ = ejd(R_t[0],R_t[1])
    C_t_diag[0],C_t_diag[1],T_C,_ = ejd(C_t[0],C_t[1])

    step=0.001;cond_a=[];cond_b=[];gamma_a=[];gamma_b=[]
    a_bad=[]
    for i in range(len(interv_a)): #
        a_test2 = np.arange(interv_a[i][0],interv_a[i][1],step)
        for j,a in enumerate(a_test2):
            cond = positivity_condition(R_t_diag,C_t_diag,a,-150)
            cond_a.append( cond[0]) #test, (normlt tous TRUE)
            if cond[0] == False:
                a_bad.append(a)
            gamma_a.append(a*R_t[0][0,1]+(1-a)*R_t[1][0,1])
    
    b_bad=[]        
    for i in range(len(interv_b)): #
        b_test2 = np.arange(interv_b[i][0],interv_b[i][1],step)
        for j,b in enumerate(b_test2):        
            cond = positivity_condition(R_t_diag,C_t_diag,150,b)
            cond_b.append(cond[1]) #test, (normlt tous TRUE)
            if cond[1] == False:#
                b_bad.append(b)
            gamma_b.append(b*R_t[0][0,1]+(1-b)*R_t[1][0,1])
            
   
    polinsar_plot_cu(mat_covariance_norm)
    plt.plot(np.angle(gamma_a),np.abs(gamma_a),'r.')
    plt.plot(np.angle(gamma_b),np.abs(gamma_b),'g.')
 
      
    
    b_debug=b_test2[-9]
    
    
    #extraction matrice interferometrique à partir de la 
    #matice de covariance en coordonné MPMB
    #creation masque
    Mask=np.zeros((3*Na,3*Na))
    for i in range(0,Na*3):
        for j in range(0,Na*3):
            Mask[i,j]=np.mod(i+j,2)
    
    """
    print "Interv numerique a :",(min(a_valable),max(a_valable))
    print "Interv a par alpha :",interv_a
    print "Interv numerique b :",(min(b_valable),max(b_valable))
    print "Interv b par alpha :",interv_b
    """
    
    
    #L=np.zeros((len(b_bad),5));L_2=np.zeros((len(b_bad),5))
    L=np.zeros((len(b_test2[60::]),5));L_2=np.zeros((len(b_test2[60::]),5))
    #b_debug=b_bad[0]
    for i,b_debug in enumerate(b_test2[60::]):
        L_C=npl.eig((1-b_debug)*C_t_diag[0]-b_debug*C_t_diag[1])[0]
        L_R=npl.eig(b_debug*R_t_diag[0]+(1-b_debug)*R_t_diag[1])[0]
        L[i,:]=np.hstack((np.real(L_R),np.real(L_C)))
        #print "vap en diagonalisé",L
        
        L_C_2=npl.eig((1-b_debug)*C_t[0]-b_debug*C_t[1])[0]
        L_R_2=npl.eig(b_debug*R_t[0]+(1-b_debug)*R_t[1])[0]
        L_2[i,:]=np.hstack((np.real(L_R_2),np.real(L_C_2)))
        #print "vap en normal",L_2
    """    
    plt.figure(1)
    plt.hold(True)
    plt.grid()
    for i in range(L.shape[1]):
        plt.plot(L_2[:,i],'-*')
    plt.hold(False)
    
    plt.figure(2)
    plt.hold(True)
    plt.grid()
    for i in range(L.shape[1]):
        plt.plot(L[:,i],'-*')
    plt.hold(False)    
    """
    
    #On se balade sur les (a,b) pour optimiser le critère     
    """    
    crit = np.zeros((len(a_valable),len(b_valable)))
    for i,a in enumerate(a_valable):
        for j,b in enumerate(b_valable):
            crit[i,j]=criterion(a*R_t[0]+(1-a)*R_t[1],b*R_t[0]+(1-b)*R_t[1])
    Opt=np.argmin(crit)
    """
    ####Test condition de bornitude de R1 (et R2)   
    #pour a
    """
    a_valable=[]
    a_valable_final=[]
    idx_a_valable_final=[]
    a_test=np.arange(0.9,1.1,0.0001)
    tab_gamma_a=np.zeros(np.size(a_test),dtype=complex)
    tab_gamma_valable_a=[]
    
    for i,a in enumerate(a_test):
        Cond,gamma=bounded_condition(R_t,a)       
        tab_gamma_a[i]=gamma
        if Cond==True:
            a_valable.append(a)
            tab_gamma_valable_a.append(gamma)
            
    for i,a_2 in enumerate(a_valable):
        Cond_2 = positivity_energy_condition(C_t,a_2)
        if Cond_2 == True:
            a_valable_final.append(a_2)
            idx_a_valable_final.append(i)
            
    #pour b
    b_valable=[]
    b_valable_final=[]
    idx_b_valable_final=[]
    b_test=np.arange(0.9,1.1,0.0001)
    tab_gamma_b=np.zeros(np.size(b_test),dtype=complex)
    tab_gamma_valable_a=[]
    
    for i,b in enumerate(b_test):
        #bounded condition car la cond sur les R_t est pareille pour a et b
        Cond,gamma=bounded_condition(R_t,b)       
        tab_gamma_b[i]=gamma
        if Cond==True:
            b_valable.append(b)
            tab_gamma_valable_a.append(gamma)
            
    for i,b_2 in enumerate(b_valable):
        Cond_2 = positivity_energy_condition2(C_t,b_2)
        if Cond_2 == True:
            b_valable_final.append(b_2)
            idx_b_valable_final.append(i)
            
    
    polinsar_plot_cu(covar)
    #plt.plot(np.angle(gamma1),np.abs(gamma1),'co')
    #plot(np.angle(tab_gamma_valable),np.abs(tab_gamma_valable),'rx')
    plt.plot(np.angle(tab_gamma_a[idx_a_valable_final]),np.abs(tab_gamma_a[idx_a_valable_final]),'r.')
    plt.plot(np.angle(tab_gamma_b[idx_b_valable_final]),np.abs(tab_gamma_b[idx_b_valable_final]),'b.')
    """
    ####
    




    
    """
    #Test du calcul des vap de MPMV sur differentes région
   
    s_tab=np.zeros((len(range(S,tropi35.nrange-S,S)),len(range(S,tropi35.nazi-S,S)),4))
    s_lign=np.zeros(4)
   
    for m,j0 in enumerate(range(S,tropi35.nrange-S,S)):
        for n,i0 in enumerate(range(S,tropi35.nazi-S,S)):
           
            data=tropi35.get_data_rect(i0,i0+S,j0,j0+S);
            data_norm=normalize_MPMB(data,Na)  
            #covar1=tropi35.get_covar_rect(il0,il1,ic0,ic1)
            covarn1=tropi35.get_covar_norm_rect(i0,i0+S,j0,j0+S)
            covarn1=covarn1/covarn1[0,0]   
            P_covar1=p_rearg(covarn1,Np,Na)    
            u,s_tab[m,n,:],v=npl.svd(P_covar1)
            s_lign=np.vstack((s_tab[m,n,:],s_lign))
            print("i0 = {0}  j0 = {1}".format(i0,j0))
    """        
    
    
    #A ADAPTER A TOMOSAR
    """
    critere='hh-hv'
    phig,psi = polinsar_calcul_phig_psi(covar,critere)
    test=tropi35.rvogsb_estime_phig_hv_rect(1275,1518,1205,1487)
    tropi35.rvogsb_bcr_rect(2402,2605+1,1312,1616+1)
    tropi35.rvogsb_attenuation_polar(2402,2605+1,1312,1616+1) 
    """
    #polinsar_plot_cu(covar)
    #polinsar_plot_cu_orientation(covar)
    #print 'Ecart type sur zg pour 100 pixels', np.sqrt(np.real(tropi35.rvogsb_bcr_rect(2402,2605+1,1312,1616+1)[0])/100)
    #print 'Ecart type sur hv pour 100 pixels', np.sqrt(np.real(tropi35.rvogsb_bcr_rect(2402,2605+1,1312,1616+1)[1])/100)
    
    #test=tropi35.rvogsb_image(0,40,0,40)
    #img=pymela.Image("tropi0403-405-phig",mode='w')
#    img.set_data(test[0],force_write=True)
#    img.close()
#    img=pymela.Image("tropi0403-405-hv",mode='w')
#    img.set_data(test[1],force_write=True)
#    img.close()





    #test diagonalisation commune
    """    
    tvol,tground,x1,x2=tropi35.polinsar_estime_tvol_tground(covar,[1000,1000])
    D1,D2,A,L=ejd(tvol,tground)
    """
    #Test rearrangement operator
    """A=np.array([(1,2,3,4,5,0),(6,7,8,9,10,0),(11,12,13,14,15,0),(16,17,18,19,20,0),(21,22,23,24,25,0),(42,43,45,46,45,42)])  
    B=np.vstack((np.hstack((A,A+1)),np.hstack((A-1,A+2))))
    m1=n1=2
    m2=n2=3    
    PA=vec(A,m1,m2,n1,n2)"""
 
    """
    #Carte hv phig    
    im_hv,im_phig=tropi35.rvogsb_image(200,2000,200,2000)
    #sauvegarde
    img=pymela.Image("hv_tropi0403-405-hv",mode='w')
    img.set_data(im_hv[0],force_write=True)
    img.close
    img=pymela.Image("hv_tropi0403-405-phig",mode='w')
    img.set_data(im_phig[0],force_write=True)
    img.close
    #plot
    figure()
    plt.imshow(im_hv)
    plt.colorbar()
    plt.title('Carte de hv') 
    figure()
    plt.imshow(im_phig)
    plt.colorbar()
    plt.title('Carte de zg')
    """
    #pylab.show()
    #im_phig,im_hv=tropi35.rvogsb_image(200,250,200,250,10,10)
    #image_phig=pymela.Image('tropi403_405_inversion_Phig.inf',mode="w",size=(300,300),dtype=float)
    #image_phig.open()
    #image_phig[:,:]=im_phig[:,:]
    #image_phig.close()
    #image_hv=pymela.Image('tropi403_405_inversion_hv.inf',mode="w",size=(300,300),dtype=float)
    #image_hv.open()
    #image_hv[:,:]=im_hv[:,:]
    #image_hv.close()

    #tvol=2*np.eye(3)
    #tground=diag((43.8,16.16,11.24))
    
if __name__ == "__main__":
    #matrice test"     
    #A=np.array([(1,2,3,4,5,0),(6,7,8,9,10,0),(11,12,13,14,15,0),(16,17,18,19,20,0),(21,22,23,24,25,0),(42,43,45,46,45,42)])  
    #B=np.vstack((np.hstack((A,A+1)),np.hstack((A-1,A+2))))      
        
    imageNameMhh1='/home/capdessus/data1/TROPISAR/Paracou/tropi0403/tropi0403_in_0402_Hh_com.inf'
    imageNameMhh2='/home/capdessus/data1/TROPISAR/Paracou/tropi0405/tropi0405_in_0402_Hh_com.inf'
    imageNameHa1='/home/capdessus/data1/TROPISAR/Paracou/alt_ambig/tropi0403proj_0405proj_traj0405_Ha.1'
    listImageNamehh=[imageNameMhh1,imageNameMhh2]
    listImageNameHa=[imageNameHa1]
  
    np.set_printoptions(precision=2)
    sigma=0.4    
    Na=2 #nbre d'antennes
    Np=3 #nbre polar
    j=0;#Indice de Baseline²
    Ki=2#nmbre de SM
    
    #Coordonnées de la fenêtre d'analayse
    il0,il1,ic0,ic1=za.zone_analyse('ZP2')
    pos_center=[(il1+il0)/2,(ic1+ic0)/2]
    tropi35=TomoSARDataSet(Na,listImageNamehh,listImageNameHa,sigma)
  
    #classique
    data=tropi35.get_data_rect(il0,il1,ic0,ic1);
    #mat_covariance=tropi35.get_covar_rect(il0,il1,ic0,ic1)
    mat_covariance_norm=tropi35.get_covar_rect(il0,il1,ic0,ic1)
    #Tebaldini    
    data_teb=tropi35.get_data_teb_rect(il0,il1,ic0,ic1)
    
    #data_norm=normalize_MPMB(data,Na)      
    W=tropi35.get_W_norm_rect(il0,il1,ic0,ic1)
    W=W/W[0,0]
    ## T SURE ??? cette noramlisation POUE 
    #mat_covariance_norm = mat_covariance_norm/mat_covariance_norm[0,0]
    P_W = p_rearg(W,Np,Na)    
    u,s,v = npl.svd(P_W)    
    #test sm_separation  
    R_t,C_t = sm_separation(W,Np,Na)  
    #test search_space_definition
    interv_a,interv_b,mat_cond,alpha = search_space_definition(R_t,C_t,Na)  
    
    
    ###########################################################################
    #chargement données synthétiques à partir des paramètres biomass 
    param= lp.load_param('DB_0')
    taille = (il1-il0)*(ic1-ic0)
    tropi35.get_data_synth(nb_echant = taille)
    
    