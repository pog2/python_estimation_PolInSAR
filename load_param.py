# -*- coding: utf-8 -*-
"""Chargement de configuration prédéfinies stockées dans une classe param
"""

import sys   
import numpy as np
import RVoG_MB as mb


def load_param(name=''):
    
    if name =='DB_0':                    
        A = 0.95
        E = 200
        X = 0.2
        N = 100
        k_z = [0.1,0.15]
        #Dualb baseline        
        T_vol,T_ground = mb.Tv_Tg_from_A_E_X(A,E,X)
        gammat=np.ones((3,3))
        k_z=k_z
        #Na=len(k_z)+1
        theta=45*np.pi/180
        sigma_v=0.0345
        h_v=30
        z_g = 4
        param = mb.param_rvog(N=N,k_z=k_z,theta=theta,T_vol=T_vol,
                              T_ground=T_ground,h_v=h_v,z_g=z_g,
                              sigma_v=sigma_v,mat_gamma_t=gammat)
        return param
        
    if name =='DB_1':                    
        A = 0.95
        E = 200
        X = 0.2
        N = 100
        k_z = [0.1,0.15]
        #Dualb baseline        
        T_vol,T_ground = mb.Tv_Tg_from_A_E_X(A,E,X)
        gammat=np.array([[1,0.7,0.8],[1,1,0.8],[1,1,1]])
        k_z=k_z
        #Na=len(k_z)+1
        theta=45*np.pi/180
        sigma_v=0.0345
        h_v=30
        z_g = 4
        param = mb.param_rvog(N=N,k_z=k_z,theta=theta,T_vol=T_vol,
                              T_ground=T_ground,h_v=h_v,z_g=z_g,
                              sigma_v=sigma_v,mat_gamma_t=gammat)
        return param
        

    if name =='DB_3':                    
        A = 0.4
        E = 200
        X = 0.2
        N = 100
        k_z = [0.1,0.15]
        #Dualb baseline        
        T_vol,T_ground = mb.Tv_Tg_from_A_E_X(A,E,X)
        gammat=np.ones((3,3))
        k_z=k_z
        #Na=len(k_z)+1
        theta=45*np.pi/180
        sigma_v=0.0345
        h_v=30
        z_g = 4
        param = mb.param_rvog(N=N,k_z=k_z,theta=theta,T_vol=T_vol,
                              T_ground=T_ground,h_v=h_v,z_g=z_g,
                              sigma_v=sigma_v,mat_gamma_t=gammat)
        return param
       
    else:
        print 'Nom de configuration non reconnue'
         
         
if __name__ =='__main__':
    print 'load_param'
        
        