# -*- coding: utf-8 -*-

from __future__ import division
import tomosar_synth as tom
import numpy as np
import numpy.linalg as npl
import scipy.linalg as scl
import matplotlib.pyplot as plt
import RVoG_MB as mb
import load_param as lp
plt.ion()

def plot_converg(vec_Crit,crit_vrai=None,**kwargs):
    """Plot l'évolution du critère *vec_Crit*.
    
    **Entrées**
        * *vec_Crit* : critère à tracer
        * *crit_vrai* : valeur vraie (si connue)
        * *\**kwargs* : dictionnaire d'arguments optionnels
        
            * *xscale,yscale* : echelle pour axes x et y ('log', 'linear' ou 'symlog')
            * *mode*          : 'diff' : trace :math:`|vec_{Crit} - crit_{vrai}|^2`
    
    **N.B** : Filtre les valeurs 1e14 (~valeurs infinie pour :func:`mlogV_zg_known <estimation.mlogV_zg_known>`)"""
    
    Idx = np.where(np.array([vec_Crit[i]==1e14 for i in range(len(vec_Crit))]))[0]
    vec_f = np.array(vec_Crit)
    #plt.plot dont display nan
    vec_f[Idx] = np.nan     
    
    if  kwargs.has_key('xscale'):
        xxscale = kwargs['xscale']
        if xxscale == 'log':
            #suprrimer l'element x=0 
            vec_X = range(1,1+len(vec_f))
            vec_f = vec_f[1:]
        elif xxscale == 'linear' or xxscale == 'symlog':           
            vec_X = range(len(vec_f))
        else:
            print 'Echelle X non reconnue'        
    else:
        #Dans le doute : echelle lineare
        xxscale = 'linear'
        vec_X = range(len(vec_f))

    if  kwargs.has_key('yscale'):
        yyscale = kwargs['yscale']
        if yyscale is not 'log' or yyscale is not 'linear' or yyscale is not 'symlog':
            print 'Echelle Y non reconnue'    
    else:
        #Dans le doute : echelle log
        yyscale = 'log'

    if crit_vrai is not None:
        vec_crit_vrai = np.ones(len(vec_X))*crit_vrai
        plt_crit_vrai = 1
    else:
        plt_crit_vrai = 0        
              
    if kwargs.has_key('mode'):
        if kwargs['mode'] == 'diff':
            vec_f = np.abs(vec_f-vec_crit_vrai)**2
            plt_crit_vrai = 0
            
        else:
            print 'mode non gere'
                
    plt.figure()
    plt.plot(vec_X,vec_f,'r')
    if plt_crit_vrai : plt.plot(vec_crit_vrai,'k')
    plt.xscale(xxscale)
    plt.yscale(yyscale)
    plt.tick_params(labelsize=14)
            
    if kwargs.has_key('show_inf'):
        show_inf = kwargs['show_inf']
        
        if(show_inf):
            if len(Idx)>0:                                               
                for i in range(len(Idx)):            
                    plt.axvline(Idx[i],color='m')        
            
def plot_err(vec_X,Y,err,Y_vrai=None,title='',fig=None,**kargs):
    """ Plot de l'évolution de *Y* avec l'erreur *err* associée. 
    
    **Entrée**
        * *vec_X* : *array* Nx1. donnée en abscisse
        * *Y* : *array*. Quantité a representer. Si *Y* est une matrice, trace chaque colonne séparement
        * *err* : *array*. Erreur sur chaque point de *Y*. Dois être de même taille de que *Y*
        * *Y_vrai* : valeur vraie de *Y* si connue
        * *fig* : *figure handle*. Si différent de *None*, plot sur une figure existante 
        * *\**kwargs* : argument optionnel 
        
            * *xscale,yscale* : echelle pour les axes ('log,'linear','symlog')"""
            
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}    
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['font.size'] = 16
    if fig == None:
        fig=plt.figure()
    else:
        plt.figure(fig.number)
     
    plt.errorbar(vec_X,Y,err,ecolor='r',linewidth=1.5,
                 elinewidth=1.2,capsize=5,color='r')
    if(Y_vrai != None):
        plt.hold(True)
        sze = vec_X.size
        plt.plot(vec_X,Y_vrai*np.ones(sze),'--k')
    if(kargs.has_key('xscale')):
        plt.xscale(kargs['xscale'])
    else:
        plt.xscale('linear')
    if(kargs.has_key('yscale')):
        plt.yscale(kargs['yscale'])
    else:
        plt.yscale('log')
    plt.title(title,fontsize=11,x=0.5,y=1.05)    
    plt.xlim(((vec_X[0],vec_X[-1])))    
    plt.grid()
    plt.rc('font',**font)  
    return fig

def pplot(vec_X,mat_Y,**kargs):
    """ plot de courbes.
    
    **Entrées** 
        * *vec_X* : axe des abscisses
        * *mat_y* : quantité a tracer : chaque colonne contient un signal a tracer de la taille de vec_X    
        *  *\**kargs* :
        
            * *names* : liste des noms des courbes (len(names)=mat_Y.shape[1])
            * *colorstyle* : definit le style 
                * *onecolor_blue* : tte les courbes en bleu (dégradé)
                * *onecolor_green* : tte les courbes en vert (dégradé)
                * *onecolor_orange* : tte les courbes en orange (dégradé)
    """    

    if  kargs.has_key('xscale'):        
        xxscale = kargs['xscale']
        if xxscale is not 'log' and xxscale is not 'linear' and xxscale is not 'symlog':            
            print 'Echelle X non reconnue'        
    else:
        #Dans le doute : echelle lineare
        xxscale = 'linear'        

    if  kargs.has_key('yscale'):
        yyscale = kargs['yscale']
        if yyscale is not 'log' and yyscale is not 'linear' and yyscale is not 'symlog':
            print 'Echelle Y {0} non reconnue'.format(yyscale)
    else:
        #Dans le doute : echelle log
        yyscale = 'log'
    
    #fontP=FontProperties()
    if(kargs.has_key('ax')):
        ax=kargs['ax']
    else:        
        fig=plt.figure(dpi=100)
        ax=fig.add_subplot(111)    
        
    ax.tick_params(labelsize=14)
        
    if(kargs.has_key('colorstyle')):
        if kargs['colorstyle'] == 'onecolor_blue':            
            cmap=plt.get_cmap('Blues')
        elif kargs['colorstyle'] == 'onecolor_green':            
            cmap=plt.get_cmap('Greens')
        elif kargs['colorstyle'] == 'onecolor_orange':            
            cmap=plt.get_cmap('Oranges')            
        else:
            cmap=plt.get_cmap('hsv')
            ax.semilogy(vec_X,mat_Y)    

        vec_color=np.linspace(.3,.8,mat_Y.shape[1])
        mat_color=cmap(vec_color)
    
        for i in range(mat_Y.shape[1]):
                ax.plot(vec_X,mat_Y[:,i],c=mat_color[i,:])
                ax.xscale(xxscale)
                ax.yscale(yyscale)                                    
    else:
        ax.semilogy(vec_X,mat_Y)           
        plt.xscale(xxscale)
        plt.yscale(yyscale)
        
    ax.grid('True',which='minor',ls='--',alpha=0.3)
    ax.grid('True',which='major',ls='--',alpha=0.5)
    if(kargs.has_key('ylim')):        
        ylim=kargs['ylim']
        ax.set_ylim(ylim)
    #else:
        #ax.set_ylim((10**-1,10**2))
    
    if(kargs.has_key('names')):        
        str_names=kargs['names']        
        if len(str_names) != mat_Y.shape[1]:
            print 'Attention: manque des noms pour la légende'
            return 0
        
        ax.legend(str_names,bbox_to_anchor=(1,1.11),
                   fontsize=8,ncol=len(str_names))
    
def plot_cu_sm_possible(Ups,R_t,interv_a,interv_b,\
                        ant1=0,ant2=1,title='CU',fig=None):
    """ Plot de la région de cohérence pour une baseline donnée avec les 
    cohérence possibles pour le sol et le volume,
    
    **Entrées**
        * *Ups* : matrice de covariance (base lexicographique)
        * *R_t* : liste. contient les deux matrices structure (SKP)
        * *interv_a* : intervalle possible pour R_t[0] 
        * *interv_b* : intervalle possible pour R_t[1] 
        * *ant_i* : indice de l'antenne 
        * *title* : titre du plot
        * *fig* : figure handle. si différent de None, plot sur une figure dejà ouverte"""          
    
    if ant1 == ant2:
        print 'Dans plot_cu_sm_possible, Attention Ant1 = Ant2'
    #plot de la région de cohérence avec les gamma correspondant à 
    #des sslutions definies positives
    nb_pts = 20
    a_possible = np.linspace(interv_a[0][0],interv_a[0][1],nb_pts)
    b_possible = np.linspace(interv_b[0][0],interv_b[0][1],nb_pts)
    gamma_a = []
    gamma_b = []
    
    for i_a,a in enumerate(a_possible):
         gamma_a.append(a*R_t[0][ant1,ant2]+(1-a)*R_t[1][ant1,ant2])
    for i_b,b in enumerate(b_possible):
         gamma_b.append(b*R_t[0][ant1,ant2]+(1-b)*R_t[1][ant1,ant2])
         
    if fig != None:
        plt.figure(fig.number)
    else:
        fig = plt.figure()
    plt.hold(True)    
    covarn = Ups

    T1=covarn[:3,:3]
    omega = covarn[3*(ant1):3*(ant1+1),3*(ant2):3*(ant2+1)]
    #tracer plusieurs cohérences obtenues de manière alléatoire
    k=np.random.randn(1000,3) + 1j*np.random.randn(1000,3)
    power = ((k.dot(T1))*(np.conj(k))).sum(axis=1)
    interf = ((k.dot(omega))*(np.conj(k))).sum(axis=1)/power
    plt.polar(np.angle(interf),abs(interf),'c.')

    #Tracé des gamma_a et gamma_b possibles
    plt.polar(np.angle(gamma_a),np.abs(gamma_a),'r.',label='g_a possible')
    plt.polar(np.angle(gamma_b),np.abs(gamma_b),'g.',label='g_b possible')
    #plt.text(1.,1.2,title)
    return fig 

def plot_rc_mb(Ups):
    """Plot de toutes les regions de cohérence (multibaseline)
    
    Plot la région de cohérence pour chaque baseline

    **Entrées**
        * *Ups* : Matrice de covariance (base lexicographique)
    """
    
    Na = int(np.floor(Ups.shape[0]/3))
    Nb = mb.get_Nb_from_Na(Na)
    
    k=np.random.randn(2000,3) + 1j*np.random.randn(2000,3)
    W=k.T
    #W=W/np.diag(W.T.conj().dot(np.eye(3).dot(W)))
    mat_coherence = np.zeros((k.shape[0],Nb),dtype='complex')    
    for p in range(Nb):
        idx2D = mb.get_idx_dble_from_idx_mono(p,Na)
        Ti = Ups[3*idx2D[0]:3*idx2D[0]+3,3*idx2D[0]:3*idx2D[0]+3]
        Tj = Ups[3*idx2D[1]:3*idx2D[1]+3,3*idx2D[1]:3*idx2D[1]+3]
        Om_ij = Ups[3*idx2D[0]:3*idx2D[0]+3,3*idx2D[1]:3*idx2D[1]+3]
        #mat cohere blanchie
        Om_w_ij = npl.inv(scl.sqrtm(Ti)).dot(Om_ij.dot(npl.inv(scl.sqrtm(Tj))))
        mat_coherence[:,p] = np.diag(W.T.conj().dot(Om_w_ij.dot(W)))/np.diag(W.T.conj().dot(np.eye(3).dot(W)))
        
    f=plt.figure()
    ax = f.add_subplot(111,polar=True)
    p=plt.polar(np.angle(mat_coherence),np.abs(mat_coherence),'.')
    ax.set_rmax(1.1)
        

def plot_biais_variance_err(varia_taille_echant,X_moy,X_var,N_real,\
                            title_moy,title_var,X_vrai=None,varX_vrai=None,
                            ylab=None,figm=None,figv=None):
    """Plot de biais et variance"""                            
    N_real = int(N_real)
    #construction intervalle de confiance    
    #erreur sur la moyenne
    err_moy=2*np.sqrt(X_var)*1/np.sqrt(N_real)    
    #erreur sur la var
    err_var=2*X_var*np.sqrt(2/(N_real-1))  
    
    plot_err(varia_taille_echant,X_moy,err_moy,N_real,X_vrai=X_vrai,
             title_moy=title_moy,fig=figm)
    plot_err(varia_taille_echant,X_var,err_var,N_real,X_vrai=varX_vrai,
             title_moy=title_moy,fig=figv)


def plot_moy(vec_X,Y,Y_vrai=None,title='',**kwargs):
    """Pot de moyenne """
    
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}    
    plt.rcParams['axes.labelsize']=25
    plt.rcParams['font.size']=16
    plt.figure()
    plt.plot(vec_X,Y,linewidth=1.5)
    if(Y_vrai != None):        
        plt.hold(True)
        sze = vec_X.size
        plt.plot(vec_X,Y_vrai*np.ones(sze),'--k')
        plt.hold(False)
    plt.title(title)
    plt.xlim(((vec_X[0],vec_X[-1])))    
    plt.grid()
    plt.rc('font',**font)  
                          

    
def subplot_cu_sm_possible(mat_covariance_norm,\
                            R_t_1,interv_a_1,interv_b_1,ant1_1,ant2_1,\
                            R_t_2,interv_a_2,interv_b_2,ant1_2,ant2_2,\
                            title=''):    
    """ Plot de la région de cohérence pour une baseline donnée avec les 
    cohérence possibles pour le sol et le volume, ceci pour deux baselines
    
    **Entrées**
        * *mat_covariance_norm* : matrice de covariance 
        * *R_t* : liste. contient les deux matrices structure (SKP)
        * *interv_a* : intervalle possible pour R_t[0] 
        * *interv_b* : intervalle possible pour R_t[1] 
        * *ant_i* : indice de l'antenne 
        * *title* : titre du plot
        * *fig* : figure handle. si différent de None, plot sur une figure dejà ouverte"""          

    figolu = plot_cu_sm_possible(mat_covariance_norm,R_t=R_t_1,interv_a=interv_a_1,\
                                 interv_b=interv_b_1,ant1=ant1_1,\
                                 ant2=ant2_1,title=title,fig=None)
                                 
    plot_cu_sm_possible(mat_covariance_norm,R_t=R_t_2,interv_a=interv_a_2,\
                        interv_b=interv_b_2,ant1=ant1_2,\
                        ant2=ant2_2,title=title,fig=figolu)
    
                        


    
def plot_vp(R_t,C_t,interv_a,interv_b,alpha):
    """ Plot des valeurs propres des R_t et C_t"""
    #Nombre de SM
    Ki=2   
    R_t_diag=[None]*Ki # Ki vecteurs (de dimensiosn Na) contenant les vap        
    C_t_diag=[None]*Ki 
    R_t_diag[0],R_t_diag[1],_,_ = tom.ejd(R_t[0],R_t[1]) #ndice ~ k ~ num du SM 
    C_t_diag[0],C_t_diag[1],_,_ = tom.ejd(C_t[0],C_t[1])      
    nb_pts = 50
    mat_vap = np.zeros((50,5))
    
    if len(interv_a) == 1 and len(interv_b) == 1:    
        # Solution def positive
        a_possible = np.linspace(interv_a[0][0],interv_a[0][1],nb_pts)
        a_plot = a_possible
        for idx_a,a in enumerate(a_possible):             
            if interv_a[0][0] > interv_b[0][1]:
                # a > b <=> min(a)>max(b)                
                R1 = a*R_t_diag[0]+(1-a)*R_t_diag[1]
                C2 = (-(1-a)*C_t_diag[0]+a*C_t_diag[1])            
            elif interv_a[0][1] < interv_b[0][0]:
                # a < b <=> max(a)>min(b)
                R1 = a*R_t_diag[0]+(1-a)*R_t_diag[1]
                C2 = (1-a)*C_t_diag[0]-a*C_t_diag[1]
            mat_vap[idx_a,:] = np.hstack((np.diag(R1),np.diag(C2)))        
    
    elif len(interv_a) == 0 and len(interv_b) == 0:
        
        #On ne teste que des a>b
        a_test = np.linspace(0,2,nb_pts)
        a_plot = a_test
        #b_test = np.linspace(0,1,nb_pts)
        for idx_a,a in enumerate(a_test):                         
            # a > b                
            R1 = a*R_t_diag[0]+(1-a)*R_t_diag[1]
            C2 = (-(1-a)*C_t_diag[0]+a*C_t_diag[1])            
            mat_vap[idx_a,:] = np.hstack((np.diag(R1),np.diag(C2)))        
            
        """
            # Pas de solution def positive pour une matrice plot trace les colonnes
        """
    print alpha
    plt.figure()
    plt.plot(a_plot,np.real(mat_vap[:,0:]),'s-',alpha,[0]*alpha.size,'or')
    plt.grid(color='k', linestyle='-', linewidth=1.5)    
    #lineObjects = plt(a_plot,np.real(mat_vap[:,0:]))
    #plt.legend(lineObjects,('R1_1','R1_2','C1_1','C1_2','C1_3'))
    
def plot_vp_Rv_Cg(R_t,C_t,interv_a,interv_b,alpha):
    """Plot des valeurs diagonale "valeurs propres" de Rv=R2
    et Cg=C1 à partir de la diagonalisation conjointe de R1,R2
    et C1,C2"""
    #Nombre de SM
    Ki=2   
    R_t_diag=[None]*Ki # Ki vecteurs (de dimensiosn Na) contenant les vap        
    C_t_diag=[None]*Ki 
    R_t_diag[0],R_t_diag[1],_,_ = tom.ejd(R_t[0],R_t[1]) #ndice ~ k ~ num du SM 
    C_t_diag[0],C_t_diag[1],_,_ = tom.ejd(C_t[0],C_t[1])     
    Na = R_t[0].shape[0] #R_t[0] matrice de structure NaxNa
    
    nb_pts = 50
    mat_vap = np.zeros((nb_pts,Na+3),dtype='complex')        
    b_test = np.linspace(alpha[0]-1,alpha[-1]+1,nb_pts)
    a = interv_a[0][0] #correspond au sol ayant un module unitaire
    
    for idx_b,b in enumerate(b_test):             
        R2 = b*R_t_diag[0]+(1-b)*R_t_diag[1]
        C1 = 1/(a-b)*((1-b)*C_t_diag[0]-b*C_t_diag[1])                    
        mat_vap[idx_b,:] = np.hstack((np.diag(R2),np.diag(C1)))        

    fig=plt.figure()
    plt.plot(b_test,np.real(mat_vap[:,0:]),'s-',alpha,[0]*alpha.size,'or')
    vec_interv_b=np.linspace(interv_b[0][0],interv_b[0][1])
    plt.plot(vec_interv_b,[-0.5]*vec_interv_b.size,'xb')
    plt.grid(color='k', linestyle='--', linewidth=1.2)    
    
def plot_vp_Rv_Cg_nodiag(R_t,C_t,interv_a,interv_b,alpha):
    """Plot des valeurs diagonale "valeurs propres" de Rv=R2
    et Cg=C1 sans diagonalisation conjointe de R1,R2
    On plot les vp de Cg (diag conjointe) et 
    gamma_i = R_t[i][0,1]"""
    #Nombre de SM
    Ki=2   
    R_t_diag=[None]*Ki # Ki vecteurs (de dimensiosn Na) contenant les vap        
    C_t_diag=[None]*Ki 
    R_t_diag[0],R_t_diag[1],_,_ = tom.ejd(R_t[0],R_t[1]) #ndice ~ k ~ num du SM 
    C_t_diag[0],C_t_diag[1],_,_ = tom.ejd(C_t[0],C_t[1])     
    Na = R_t[0].shape[0] #R_t[0] matrice de structure NaxNa
    
    
    nb_pts = 450
    Nb_baseline = int(Na*(Na-1)/2)
    #Na*(Na-1)/2 + 3 : Na*(Na-1)/2 pour les gammm_v et 3 pour les vp de C1
    mat_vap = np.zeros((nb_pts,Nb_baseline+3),dtype='complex') 
    mat_vap_abs = np.zeros((nb_pts,Nb_baseline+3),dtype='float')
    b_test = np.linspace(alpha[0]-1,alpha[-1]+1,nb_pts)
    a = interv_a[0][0] #correspond au sol ayant un module unitaire
    ratio = np.zeros(nb_pts,dtype='complex')
    ratio_g = np.zeros(nb_pts,dtype='complex')
    r1 = np.zeros(nb_pts,dtype='float')
    r2 = np.zeros(nb_pts,dtype='float')
    coeffa =np.zeros(nb_pts,dtype='float')
    coeffa_g =np.zeros(nb_pts,dtype='float')   
    r1_g = np.zeros(nb_pts,dtype='float')
    r2_g = np.zeros(nb_pts,dtype='float')   
    gamma_Rv = np.zeros((nb_pts,Nb_baseline),dtype='complex')
    gamma_Rv_g = np.zeros((nb_pts,Nb_baseline),dtype='complex')
    
    for idx_b,b in enumerate(b_test):                    
        gamma_Rv[idx_b,:],ratio[idx_b],r1[idx_b],r2[idx_b],coeffa[idx_b] = tom.rac_def_pos(R_t,a,b)
        C1 = 1/(a-b)*((1-b)*C_t_diag[0]-b*C_t_diag[1])                           
        mat_vap[idx_b,:] = np.hstack((gamma_Rv[idx_b,:].reshape(1,3),np.diag(C1).reshape(1,3)))    
        mat_vap_abs[idx_b,:] = np.hstack((np.abs(gamma_Rv[idx_b,:]).reshape(1,3),\
                                np.real(np.diag(C1).reshape(1,3))))
        #mat_vap[idx_b,:3] = np.abs(gamma_Rv).reshape(1,3)
    vec_interv_b=np.linspace(interv_b[0][0],interv_b[0][1],nb_pts)                                
    
    for idx_b_good,b in enumerate(vec_interv_b):
        gamma_Rv_g[idx_b_good,:],ratio_g[idx_b_good],\
        r1_g[idx_b_good],r2_g[idx_b_good],coeffa_g[idx_b] = tom.rac_def_pos(R_t,a,b)
        
        
    plt.figure()
    plt.plot(alpha,[0]*alpha.size,'ok')
    cns1 = 1-(np.abs(mat_vap[:,0])**2+np.abs(mat_vap[:,1])**2+np.abs(mat_vap[:,2])**2)+\
                2*np.real(mat_vap[:,0]*np.conj(mat_vap[:,1])*mat_vap[:,2])
    
    plt.plot(b_test,cns1,'c-*') #PLOT DE Real(g_12²g_13*) dans le cas Na=3
    
    plt.plot(vec_interv_b,[-0.5]*vec_interv_b.size,'xb')
    plt.grid(color='k', linestyle='--', linewidth=1.2)    
    
    plt.figure()
    plt.hold(True)
    plt.polar(np.angle(ratio),np.abs(ratio),label='ratio g13/g12^2')
    plt.polar(np.angle(ratio[0]),np.abs(ratio[0]),'ok',label='b0=alpha[0]-1='+str(b_test[0]))
    plt.polar(np.angle(ratio[-1]),np.abs(ratio[-1]),'ro',label='b_fin=alpha[-1]+1='+str(b_test[-1]))
    plt.polar(np.angle(ratio_g),np.abs(ratio_g),'g',linewidth='5',label='ratio sur interv_b')
    plt.legend()
    
    plt.figure()
    plt.hold(True)
    #plt.plot(b_test,np.abs(ratio),label='module ratio')
    #plt.plot(b_test,np.angle(ratio),label='phase ratio')
    plt.grid(True)    
    plt.plot(b_test,r1,label='rac1')
    plt.plot(b_test,r2,label='rac2')
    plt.plot(b_test,np.abs(mat_vap_abs[:,0]**2),'r-',linewidth='1.5',label='|g_12|^2')  
    plt.plot(b_test,coeffa,'k',label='coeff a du polynome')
    #plt.plot(vec_interv_b,np.abs(ratio_g),'--g',linewidth='3',label='module ratio good')
    #plt.plot(vec_interv_b,np.angle(ratio_g),'--g',linewidth='3',label='angle ratio good')
    plt.plot(vec_interv_b,r1_g,'--g',linewidth='3',label='rac1 good')
    plt.plot(vec_interv_b,r2_g,'--g',linewidth='3',label='rac2 good')
    
    plt.hold(False)
    plt.legend()

    plt.figure()
    plt.hold(True)
  
    plt.grid(True)    

    plt.semilogy(b_test,np.abs(mat_vap_abs[:,0]**2),'r-',linewidth='1.5',label='|g_12|^2')    
    plt.semilogy(vec_interv_b,r1_g,'--g',linewidth='3',label='rac1 good')
    plt.semilogy(vec_interv_b,r2_g,'--g',linewidth='3',label='rac2 good')
    plt.hold(False)
    plt.title('plot_log_interv_b' )
    plt.legend()
    
def plot_phiv(fig=None,ax=None):
    import estimation as es
    ima_nv,ima_phiv = es.load_nv_phiv('/data2/pascale/')
    plt.figure()
    size_phiv = ima_phiv[:].shape
    x = np.linspace(-1,1,size_phiv[0])
    y = np.linspace(-1,-1,size_phiv[1]) 
    X,Y = np.meshgrid(x,y)
    plt.imshow(np.transpose(ima_phiv[:]),origin='lower')
    plt.colorbar()
    return fig,ax
    
def plot_nv(fig=None,ax=None):
    import estimation as es
    ima_nv,ima_phiv = es.load_nv_phiv('/data2/pascale/')
    
    plt.figure()
    plt.imshow(ima_nv[:],origin='lower')
    plt.colorbar()
    
def I1(hv,sigmav,costheta):
    """ Calcul de I1. :math:`I_1=\frac{1-a}{\alpha}`"""
    alpha = 2*sigmav/costheta
    return (1-np.exp(-alpha*hv))/alpha
    
def I2(hv,sigmav,costheta,kz):
    """ Calcul de I2."""
    alpha = 2*sigmav/costheta
    return (np.exp(1j*kz*hv)-np.exp(-alpha*hv))/(1j*kz+alpha)     
    
def gammav(hv,sigmav,costheta,kz):
    """ Calcul de gammav. :math:`\gamma_v=\frac{I_2}{I_1}`"""
    return I2(hv,sigmav,costheta,kz)/I1(hv,sigmav,costheta)
    
def plot_gammav(costheta,kz,title='',fig=None):
    nbsig =25
    nbh=50        
    hv =np.linspace(0,2*np.pi/kz,nbh)
    sigmav=np.hstack((np.logspace(-3,0,nbsig),-np.logspace(-3,0,nbsig)))
    mat_gammav =np.zeros((sigmav.size,nbh),dtype='complex')
    
    for i,sig in enumerate(sigmav):
        for j,h in enumerate(hv):
            mat_gammav[i,j]=gammav(h,sig,costheta,kz)
    if fig ==None:
        fig=plt.figure()    
    else:
        plt.figure(fig.number)
    plt.polar(np.angle(mat_gammav),np.abs(mat_gammav))
    plt.title(title)
    return fig
    
def plot_mod_gamma_v(costheta,h,sig,vec_kz,title='',fig=None):
    alpha = 2*sig/costheta
    abs_gam = np.zeros(vec_kz.shape[0])
    plt.figure()
    for i,kz in enumerate(vec_kz):
        abs_gam[i]=np.abs(gammav(h,sig,costheta,kz))
    plt.plot(vec_kz,abs_gam,'rs-')
    
def plot_gammav_from_kz(vec_kz,h,sig,costheta):
    nb=vec_kz.size
    vec_gammav =np.zeros((nb,1),dtype='complex')
    for i,kz in enumerate(vec_kz):
        vec_gammav[i] =gammav(h,sig,costheta,kz)
    plt.figure()        
    plt.polar(np.angle(vec_gammav),np.abs(vec_gammav))

def pplot_monochrome_blocs(vec_X,list_matY,**kargs):
    """ Plot de tous les elements colonnes list_matY
    
    Chaque element de list_matY contient une matY (len(vec_X),nb_courbe)
    chaque bloc (element de list_maty) correspond a une couleur

    Remarque : boucle sur 3 couleur (26/05/16)
    
    **Entrées** 
        * *vec_X* : vecteur pour l'axe des abs
        * *list_matY* : liste de matrices matY. matY contient les données sur ses colonnes
        * *\**kargs* : bloc_list_names : liste de liste des noms des courbes.
                ex: bloc_list_names=[['crbe_grp1_1','crbe_grp1_2'],['crbe_grp2_1','crbe_grp2_2']]
    """
    
    nb_bloc=len(list_matY)
    cstyle=['onecolor_blue','onecolor_green','onecolor_orange']
    fig=plt.figure(dpi=100)
    axe=fig.add_subplot(111) 
    for i in range(nb_bloc):        
        pplot(vec_X,list_matY[i],colorstyle=cstyle[i%3],ax=axe)
        
    if(kargs.has_key('bloc_list_names')):                
        dlist=kargs['bloc_list_names']        
        list_names=[dlist[i][j] for i in range(len(dlist)) for j in range(len(dlist[i]))]        
        if len(list_names) != np.sum([list_matY[i].shape[1] for i in range(len(list_matY))]):
            print 'Attention manque des elements pour la légende'            
        axe.legend(list_names,bbox_to_anchor=(1,1.11),
                   fontsize=8,ncol=len(list_names))
                   

def launch():
    
    param_MB = lp.load_param('DB_0')
    #param_MB.gammat=[1,1]
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon_gt())
    W_k_norm,_ = tom.normalize_MPMB_PS_Tebald(W_k_vrai_MB,param_MB.Na)
    #Données bruitées
    #data_MB=tom.TomoSARDataSet_synth(param_MB)
    #W_k_norm = data_MB.get_W_k_norm_rect(param_MB,nb_echant,Na)
       
    Np=3
    R_t,C_t,_ = tom.sm_separation(W_k_norm,Np,param_MB.Na)
    interv_a,interv_b,mat_cond,_ = tom.search_space_definition(R_t,C_t,param_MB.Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)
    
    b=(interv_b[0][0]+interv_b[0][1])/2
    #b = interv_b[0][0]
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]    
    
    Rv=b*R_t[0]+(1-b)*R_t[1]
    vec_gm=tom.gamma_from_Rv(Rv)
    
    Npt_hv=100
    Npt_sig=500    
    vec_hv=np.linspace(5,30,Npt_hv)
    vec_sigv=np.linspace(0.01,0.1,Npt_sig)
    
    
    
if __name__ == "__main__":
    
    np.set_printoptions(linewidth=150)
    launch()    
    
    
    
    