# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:06:20 2015

@author: capdessus

Ensemble de fonctions additionnelles python
"""
from matplotlib.colors import LogNorm
import matplotlib as m
import matplotlib.ticker as ticker
import pdb
import sys
from copy import copy
import numpy as np
import numpy.linalg as npl 
import math as ma
import datetime as date
import matplotlib.pyplot as plt
import os
import RVoG_MB as mb
import platform as ptm


def infstd(vec):
    """Calcule l'ecart type de vec sans prendre en compte les infs"""
    
    Id = np.where(np.isinf(vec))
    if len(Id[0])>1:
        new_vec = vec[Id]
    else :
        new_vec = vec.copy()
    iinfstd = np.nanstd(new_vec)
    return iinfstd

def fmt(x, pos):
    """ Formatage des donnes (utilisé notamenet pour des colobar dans
    *iimshow*)"""
    
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${0} \times 10^{{{1}}}$'.format(a, b)
    
    
def save_fig(filename):
    """Sauvegarde une figure (issus d'un plot par ex.) à l'emplacement *filename*
    
    **Entrées** :
    
    * *h_fig* : header de la figure
    * *name* : chemin du fichier au format /repertoire/nom.png"""        
    
    folderpath,name = os.path.split(os.path.abspath(filename))      
    
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    plt.tight_layout()
    plt.savefig(filename,dpi=300,format='png',bbox_inches='tight',pad_inches=0.1)    

def save_txt_param(param,folder_path,name='config',showTvTg=True):
    """Sauvegarde dans un fichier texte les paramètres de la classe param
    
    **Entrées** :
    
    * *param* : object de type param_rvog
    * *h_fig* : header de la figure
    * *name*  : chemin du fichier au format /repertoire/nom.png"""        
    
    if os.path.exists(folder_path) == False:
        #creation du dossier s'il n'existe pas
        os.makedirs(folder_path)

    str_param ='N {0}\nkz_1n {1}\ntheta {2}\n'.format(str(param.N),str(param.k_z),str(param.theta))\
    +'h_v {0}\nz_g {1}\nsigma_v {2}\n'.format(str(param.h_v),str(param.z_g),str(param.sigma_v))\
    +'phig {0}\n'.format(str(param.k_z[0]*param.z_g))
    
    Nb=mb.get_Nb_from_Na(param.Na)
    str_gt=[]
    for i in range(Nb):
        idx2D = mb.get_idx_dble_from_idx_mono(i,param.Na)
        str_gt.append('gt{0}{1}={2}'.format(idx2D[0],idx2D[1],param.get_gamma_t(idx2D[0],idx2D[1])))
    str_gt = " ".join(str_gt)
    str_param = str_param+str_gt

    if showTvTg:
        str_param = str_param +'\nT_vol\n'\
        +'\n'.join(' '.join(str(cell) for cell in row) for row in param.T_vol)\
        +'\nT_ground\n'\
        +'\n'.join(' '.join(str(cell) for cell in row) for row in param.T_ground)    
        
    
    filename = os.path.join(folder_path,name+',txt')
    file_log = open(filename,'w')
    file_log.write(str_param)
    file_log.close()
 
def save_txt_simu(dict_simu,folder_path,name='config_simu'):
    """Sauve dans un fichier txt les paramètres de simu monte carl
    
    **Entrées** :
        * *dict_simu* : dictionnaire contenant les info sur la simu
	* *folder_path* : chemin du dossier
	* *name*  : nom du fichier"""        

    if os.path.exists(folder_path) == False:
        #creation du dossier s'il n'existe pas
        os.makedirs(folder_path)
    
    str_simu = '\n'.join(['{0}={1}'.format(items[0],items[1]) for items in dict_simu.items()])
    filename = os.path.join(folder_path,name+'.txt')
    file_simu = open(filename,'w')
    file_simu.write(str_simu)
    file_simu.close()
        
def iimshow(mat,vec_lin=None,vec_col=None,**kargs):
    """ Version personnalisée de la fonction imshow (module matplotlib.pyplot )
    
    Representation d'une fonction 2D Y=f(X,Y)
    **Entrées** 

        * *vec_lin* : vecteur correspondant aux valeurs de X.
        * *vec_col* : vecteur correspondant aux valeurs de Y
        * *\**kargs* : argument optionel :
            * *zlim* : tuple (*zmin*, *zmax*). fixe les bornes min et max pour l'axe z
            * *zscale* : string : 'log' ou 'linear'. Determine le type d\'echelle de l\'axe z
            * *xscale* : string : 'log' ou 'linear'. Determine le type d\'echelle de l\'axe x
            * *yscale* : string : 'log' ou 'linear'. Determine le type d\'echelle de l\'axe y
        
        * *plot_contour* : string : permet d'ajouter des tracés d'isocontours. Si *plot_contour='minmax'* : les contours espacées logarithmiquement du min(Z) à max(Z)
        * *show_min* : représente le mininum
    """
        
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}    
    plt.rcParams['axes.labelsize']=25
    plt.rcParams['font.size']=16
    
    if kargs.has_key('zlim'):
        zlim = kargs['zlim']
    else:
        zlim = (np.nanmin(mat),np.nanmax(mat))
        
    if kargs.has_key('zscale'):
        if kargs['zscale'] == 'log':            
            nnorm = LogNorm(vmin=zlim[0],vmax=zlim[1])
            mn = zlim[0]
            mx = zlim[1]
            ticks=np.logspace(np.log10(mn),np.log10(mx),10)            
        if kargs['zscale'] == 'linear':
            nnorm = None               
            ticks = None
    else:
        nnorm = None               
        ticks = None
        
    if vec_lin is None:
        vec_lin = range(mat.shape[0])
    if vec_col is None:
        vec_col = range(mat.shape[1])
    #pdb.set_trace()
    implot = plt.imshow(mat,origin='lower',norm=nnorm,aspect='auto',
                        extent=(vec_col[0],vec_col[-1],vec_lin[0],vec_lin[-1]))   
    if kargs.has_key('xscale'):        
        plt.xscale = kargs['xscale']
    if kargs.has_key('yscale'):        
        plt.yscale = kargs['yscale']

    plt.clim(vmin=zlim[0],vmax=zlim[1])           
    

    plt.colorbar(implot,format=ticker.FuncFormatter(fmt),
                 ticks=ticks)
       
    if kargs.has_key('plot_contour'):
        if kargs['plot_contour']=='minmax':            
            plt.hold(True)            
            X,Y = np.meshgrid(vec_col,vec_lin)
            CS=plt.contour(X,Y,mat,np.logspace(np.log10(np.min(mat)),np.log10(np.max(mat)),10),colors='white')
            plt.clabel(CS,inline=1,fontsize=16,fmt=ticker.FuncFormatter(fmt))
            plt.hold(False)        
        else:
            plt.hold(True)
            X,Y = np.meshgrid(vec_col,vec_lin)            
            CS = plt.contour(X,Y,mat,np.logspace(np.log10(zlim[0]),np.log10(zlim[1]),15),colors='white')            
            plt.clabel(CS,inline=1,fontsize=16,fmt=ticker.FuncFormatter(fmt))
            plt.hold(False)            
           
    if kargs.has_key('show_min'):
        plt.hold(True)
        idx_argmin=np.where(mat==np.min(mat))
        plt.plot(vec_col[idx_argmin[1]],vec_lin[idx_argmin[0]],'or')
        plt.hold(False)
            
    plt.rc('font',**font)  
    
            
def load_npy_data(dirname):
    """Charge des données au format npy stockées dans un dossier
    
    **Entrée**
        * *dirname* : emplacement du dossier
        * *ddata* : dictionnaire contenant les données. Chaque nom de clef\n
                    de ddata correspond au nom du fichier chargé \n
                    ex : ddata["hv"] contient *dirname/hv.npy*                    
    """    
    ddata={}
    for file in os.listdir(dirname):
        if file.endswith(".npy"):
            if ptm.system()=='Windows':
                ddata[file[:-4]]=np.load(dirname+'\\'+file)
            elif ptm.system()=='Linux':
                ddata[file[:-4]]=np.load(dirname+'//'+file)
    return ddata
    
def nans(shape, dtype=complex):
    """Initalisation d'un array par des nan.
    
    **Entrée** :
        * *shape* : tuple. Taille du tableau
        * *dtype* : type de données (complex par default)
    """    

    a = np.empty(shape,dtype)
    a.fill(np.nan)
    return a
    
def inf(shape, dtype=float):
    """Initalisation d'un array par des inf.
    
    **Entrée** :
        * *shape* : tuple. Taille du tableau
        * *dtype* : type de données (float par default)
    """    
    a = np.empty(shape,dtype)
    a.fill(np.inf)
    return a
    
def inv_cond(mat,cond_max=1e14,name=' ',return_type='nan',disp=True):
    """Renvoie l'inverse de la matrice 
    si son conditionnement est inferieur a cond_max
    sinon renvoie une matrice de nan

    **Entrée** :
        * *mat* : *array*. Matrice à inverser.
        * *cond_max* : contionnement au dela duquel nan est renvoyé
        * *name* : nom de la matrice (utile si *disp* = True)
        * *return_type* : type de valeur a retourner en cas de mauvais conditionnement
                            (par default nan)
        * *dtype* : type de données (float par default)    
        * *disp*  : mode verbeux    
    """
    
    if npl.cond(mat)<cond_max:
        return npl.inv(mat)
    else:
        if disp:
            print 'Matrice',name,'singulière!','Cond= ','{:.5}'.format(npl.cond(mat))
        if return_type == 'nan':
            return nans(mat.shape)
        if return_type == 'inf':          
            return inf(mat.shape)
        else:
            return nans(mat.shape)
        
def is_hermitian_num(A):   
    """Determine si A est hermitienne "numeriquement"
       i,e si ||A-A.H| < eps """
    eps=1e-3
    if(npl.norm(A-np.conj(A.T),'fro')<eps):
        return True
    else:
        return False

def estime_line_svd(z,theta_vrai):
    """ regression lineaire d'un ensemble de points dans le plan complexe.
    
    **Entrées** :
        * *z* : ensemble de points
        * *theta_vrai* : valeur vraies de theta (pour lever l'ambiguité)
        
    Auteur: Antoine Roueff
    Mdoèle de la droite : 
    sin(theta) x + cos(theta) y = beta avec z=x+1i*y
    pour retrouver le modèle y=ax+b il faut donc
    a=-tan(theta)
    """
    
    x=np.real(z)
    y=np.imag(z)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    x = x - x_mean
    y = y - y_mean
    
    u,s,_=npl.svd(np.vstack((x,y)))
    theta = ma.atan(u[1,0]/u[0,0])

    theta_possible = [theta,(np.pi-theta)]
    idx_min = np.argmin(theta_possible-theta_vrai)
    """Angle minimal entre la droite et l'axe horizontal
    (le -1 vient du fait qu'on mesure l'angle (droite,ux))
    la svd nous donne l'angle (ux,droite)"""
    theta2= -theta_possible[idx_min]
    return theta2

def strval(val,name=''):
    """Renvoie la chaine 'name=val'
 
    **Entrée** :
        * *name* : chaine de caratère (nom de la variable)
        * *val* : valeur afficher
    
    **Sortie** :
        * *ch* : chaine ->'name=val'        
    """
    
    if name != '':
        prefixe=name+'='        
    else:
        prefixe=''
        
    if type(val) == int:
        ch =prefixe+'{:d}'.format(val)   
    elif type(val) == float:
        ch=prefixe+'{:2.2f}'.format(val)
    else:
        ch=prefixe+'{:2.2f}'.format(val)

    return ch
    
def printm(A):
    """Print une matrice sans les crochets"""        
    
    np.set_printoptions(linewidth=200,precision=5)
    
    print '\n'.join('  '.join('%02.4f'%(np.real(cell))\
            +'+i'+'%02.2f'%(np.imag(cell)) for cell in row) for row in A)


def load_txt(path):
    """Lecture d'une matrice stockée dans un .txt"""
    
    f=open(path,'r')
    A=np.zeros((6,6),dtype='complex')
    strLines = f.readlines()
    for i in range(len(strLines)):
        line = strLines[i].replace('\r\n','')
        line = line.split('  ')
        for j in range(len(line)):
            idx_plus = line[j].find(' + ')            
            if idx_plus == -1:                
                #c'est un nb z = a -b*1i                
                strZ = line[j].split(' - ')            
                rZ = float(strZ[0])
                imZ=float(strZ[1].split('i')[0])
                signe = -1
            else:                
                #c'est un nb z = a +b*1i               
                strZ = line[j].split(' + ')            
                rZ = float(strZ[0])
                imZ=float(strZ[1].split('i')[0])
                signe = +1
                
            A[i,j] = rZ+1j*signe*imZ       
    return A
    
def get_date_DM():     
    """Renvoie la date sous forme jour/mois """     
    
    now = date.datetime.now()
    date_DM = str(now.day)+'_'+str(now.month)+'/'     
    return date_DM

def multi_hhist(mat_data,**kwargs):
    """Superpose les histogrammes contenus dans la mat de données *mat_data*
    Chaque colonne correspond à un histogramme a tracer
        
    **Entrées** 
        * *mat_data* : matrice contenant un histogramme a tracer sur chaque colonne
        * *kwargs* : dictionnaire d'options. contient les keys
            * *range* : fixe les bornes de l'histogramme. Si *mat_data* a plusieurs colonnes, les bornes affectent tous les histogrammes.    

    """
    
    if kwargs.has_key('range'):
        rrange=kwargs['range']
    else:
        rrange=(np.min(mat_data),np.max(mat_data))
        
    for i in range(mat_data.shape[1]):
        plt.hist(mat_data[:,i],50,alpha=0.5,range=rrange,label=str(i))
    plt.legend()
    
def hhist(vec_data,**kwargs):
    """Trace un histogramme contenu dans le vec de donnée vec_data
    
        * *vec_data* : vecteur contenant l'histogramme
        * *kwargs* : dictionnaire d'options. contient les keys
            * *range* : fixe les bornes de l'histogramme.
            * *title* : affecte un titre                    
    """
    
    if kwargs.has_key('range'):
        rrange=kwargs['range']
        
    else:
        mmin=np.percentile(vec_data,10)
        mmax=np.percentile(vec_data,90)
        rrange=(mmin,mmax)
    if kwargs.has_key('title'):
        title=kwargs['title']
    else:
        title=''
    
    plt.hist(vec_data,bins=80,range=rrange)
    plt.title(title,loc='center')

def rot_omega_sm(centre,theta,Ups,R_t):
    """Rotation de la phase interferometrique en SB
    
    Applique une rotation de centre *centre* et d'angle *theta*
    à la cohérence interferometrique *gamma*
    
    **Entrées** : 
        * *centre* : complex. centre de la rotation
        * *theta* : float. angle de rotation
        * *Ups* : matrice de covariance SB
        * *R_t* : matrice de structure (cf Tebaldini). matrice hermtienne 
                 tq R_t[i,i]=1 et R_t[0,1]=gamma
                 
    **Sortie** :
        * *Ups_rot* : matrice de covariance SB affectée de la rotation
        * *R_t_rot* : matrice de structure affectée de la rotation
    """
    R_t_rot = copy(R_t)
    for i in range(2):
        gamma = complex(R_t[i][0,1])
        gamma_rot = rot_cpx(centre,theta,gamma)        
        R_t_rot[i][0,1] = gamma_rot
        R_t_rot[i][1,0] = gamma_rot.conj()
    Omega= Ups[:3,3:]
    Omega_rot = rot_cpx(centre,theta,Omega)
    Ups_rot = Ups
    Ups_rot[:3,3:]=Omega_rot
    Ups_rot[3:,:3]=Omega_rot.T.conj()
    
    return Ups_rot,R_t_rot
    
def rot_cpx(centre,angle,z):
    """Rotation dans le plan complexe de centre *centre* 
    et d'angle *angle*
    
    **Entrées** : 
        * *centre* : complex. centre de la rotation
        * *angle* : float. angle de rotation
        * *z*     : nbre complexe a rotationner
        
    **Sortie** :
        * *z_rot* : complex.
        """
    
    if isinstance(z,complex)==True:
        z_rot = (z-centre)*np.exp(1j*angle)+centre
    else:
        z_rot = (z-centre*np.ones(z.shape))*np.exp(1j*angle)+centre*np.ones(z.shape)
        
    return z_rot
    
def gene_txt_of_load(dirname):
    """Genere un fichier txt contenant sur chaque ligne
    nomdufichier = np.load(dirname/nomdufichier.npy)
    
    nomdufichier correspont a tous les fichiers contenus dans dirname
    """
    str_list=[]
    list_name=[]
    print 'Yolo Debut'
    for file in os.listdir(dirname):
        if file.endswith(".npy"):
            str_list.append(file[:-4]+'=np.load(dirname+'+'\"/\"'+'+\"'+file+'\")')
            list_name.append(file[:-4])
    print '\n'.join(str_list)
    print ','.join(list_name)
    return str_list
    

def isposit(A):
    """Renvoie 1 si A est (semi) définie positive
    <=> ttes ses vap sont positives (strict)"""
    
    vec_lmbda,_=npl.eig(A)
    if np.sum(vec_lmbda>=0)==A.shape[0]:
        return 1
    else:
        return 0
    list.extend()

def KroDelta(a,b):
    """Renvoie 1 si a=b 0 sinon"""
    
    if (a==b):
        return 1
    else:
        return 0
        

    
