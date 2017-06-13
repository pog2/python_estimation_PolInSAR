# -*- coding: utf-8 -*-
"""
Exemples d'utilisation.
"""
import matplotlib.pyplot as plt
import numpy as np

import load_param as lp
import tomosar_synth as tom
import plot_tomo as pt
import estimation as e 

def example_plot_tomo():
    Np = 3 #Fullpolar
    
    param_MB = lp.load_param('DB_0')
    param_MB.N = 1000
    W_k_vrai_MB = tom.UPS_to_MPMB(param_MB.get_upsilon())

    data_tomo=tom.TomoSARDataSet_synth(param_MB)
    
    W_k_noise = data_tomo.get_W_k_rect(param_MB,nb_echant=1000)
    Ups_noise = tom.MPMB_to_UPS(W_k_noise)    
    
    W_k_MB = W_k_noise
    
    R_t_MB,C_t_MB,G_MB = tom.sm_separation(W_k_MB,Np,param_MB.Na)
    interv_a_MB,interv_b_MB,Cond,alpha = tom.search_space_definition(R_t_MB,C_t_MB,param_MB.Na) 

    
    Ups = tom.UPS_to_MPMB(W_k_vrai_MB)
    pt.plot_rc_mb(Ups)
    pt.plot_rc_mb(tom.MPMB_to_UPS(W_k_noise))
    

    pt.plot_cu_sm_possible(Ups_noise,R_t_MB,interv_a_MB,interv_b_MB,0,1,'kz12')
    pt.plot_cu_sm_possible(Ups_noise,R_t_MB,interv_a_MB,interv_b_MB,0,2,'kz13')
    pt.plot_cu_sm_possible(Ups_noise,R_t_MB,interv_a_MB,interv_b_MB,1,2,'kz23')
    
def exemple_estimateur_SKP():
    
    param = lp.load_param(name='DB_1')        
    W_k_vrai = tom.UPS_to_MPMB(param.get_upsilon_gt())
    data = tom.TomoSARDataSet_synth(param)
    Wnoise = data.get_W_k_rect(param,10**6)
    
    #W = W_k_vrai
    W = Wnoise
    W_norm,_ = tom.normalize_MPMB_PS_Tebald(W,param.Na)
    
    bvrai=tom.b_true(param)
    hv,sigv,vec_gt,bopt = e.estim_ecart_ang_scal(W,param,
                                                 critere='J2',
                                                 zg_connu=param.z_g,
                                                 U0=np.array([param.h_v,param.sigma_v]),
                                                 b0=bvrai)
   
    print 'b_vrai {0} b {1}'.format(bvrai,bopt)
    print 'hv_vrai {0} hv {1}'.format(param.h_v,hv)
    print 'sig_vrai {0} sig {1}'.format(param.sigma_v,sigv)
    print 'vec_gt_vrai {0}\n vec_gt {1}'.format(param.get_gtlist(),vec_gt)
    

def exemple_SKP():
    
    Np=3
    param=lp.load_param(name='DB_1')   
    param.z_g = 0
    data = tom.TomoSARDataSet_synth(param)
    Wnoise = data.get_W_k_rect(param,10**3)
    Ups_noise = tom.MPMB_to_UPS(Wnoise)
    
    W = Wnoise
    Ups = param.get_upsilon()
    W = tom.MPMB_to_UPS(Ups)
    #W = tom.UPS_to_MPMB(param.get_upsilon())
    W_norm,_ = tom.normalize_MPMB_PS_Tebald(W,param.Na)
        
    #SKP
    R_t,C_t,_ = tom.sm_separation(W_norm,Np,param.Na)
    
    interv_a,interv_b,_,_,_ = tom.search_space_definition_rob(R_t,C_t,param.Na)
    interv_a,interv_b = tom.ground_selection_MB(R_t,interv_a,interv_b)

    #choix du a 
    g_sol1 = interv_a[0][0]*R_t[0][0,1]+(1-interv_a[0][0])*R_t[1][0,1]
    g_sol2 = interv_a[0][1]*R_t[0][0,1]+(1-interv_a[0][1])*R_t[1][0,1]    
    g_sol_possible = np.array([g_sol1,g_sol2])
    a = interv_a[0][np.argmax(np.abs(g_sol_possible))]        

    b = 0.5*(interv_b[0][0]+interv_b[0][1])
    
    gsol,gvol = tom.gamma_a_b(R_t,a,b,ant1=0,ant2=1)
    
    pt.plot_cu_sm_possible(Ups,R_t,interv_a,interv_b,0,1,'kz12')
    plt.hold(True)
    plt.polar(np.angle(gsol),np.abs(gsol),'ro',markersize=7)
    plt.polar(np.angle(gvol),np.abs(gvol),'go',markersize=7)
    plt.hold(False)

    
if __name__ == '__main__':
    #example_plot_tomo()
    #exemple_estimateur_SKP()
    exemple_SKP()