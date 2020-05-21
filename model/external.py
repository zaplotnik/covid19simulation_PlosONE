#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 04:35:48 2020

@author: ziga
"""

import numpy as np
import matplotlib.pyplot as plt
from statistics_module import * 
from scipy.stats import *
from scipy.optimize import minimize

def plot_distribution(connection_other_max,maxc_other,theta,N,day):
    
    tp = np.zeros(maxc_other,dtype=np.int32)
    for i in range(maxc_other):
        tp[i] = (connection_other_max==i).sum()
    
    fig=plt.figure(day)
    plt.loglog(np.arange(0,maxc_other),tp,'ko-',label=r"$\theta=${0:7.3f} (mean = {1:4.1f} outer contacts /person /day)".format(\
                theta,np.sum(1.*connection_other_max)/N))
    plt.xscale('log')
    plt.xlim([0.9,maxc_other])
    plt.ylim([1,2*10**6])
    plt.xlabel(r"Number of contacts $x$")
    plt.ylabel(r"Number of people with $x$ contacts")
    ticks_num= [1,2,3,4,5,6,8,11,16,21,31,51,71,101]
    str_ticks_num= [str(tck-1) for tck in ticks_num]
    plt.xticks(ticks_num,str_ticks_num)
    plt.grid()
    plt.title("Contacts outside household/care clusters")
    props = dict(boxstyle='round', facecolor='red', alpha=0.4)  
    # plt.text(1.1,2000,"Average number of outer contacts\n per person per day: {:5.2f}".format(\
    #     np.sum(1.*connection_other_max)/N),bbox=props)
    # plt.text(1.1,100,"Average number of family contacts\n per person per day: {:5.2f}".format(\
    #     np.sum(1.*connection_family_max)/N-1),bbox=props)
    # plt.text(0.34,2,"{0} persons have 0 contact per day\n{1} persons have 1 contact per day\n{2} persons have 5-6 contacts per day".format(\
    #         int(tp[0]),int(tp[1]),int(tp[5])),bbox=props)  
    fig.tight_layout(rect=(0,0,1,1),h_pad=0.1,w_pad=0.1,pad=0.1)
    plt.legend()
    plt.savefig("kontakti_day_{:02d}.png".format(day),dpi=300)
    plt.clf()
    
    
    
def compute_parameter(v_50=None,v_025=None,v_975=None,rnd_cdf=None):
    # x0 = [(v_975-v_025)/v_50,v_50]
    x0 = [0.3,v_50]
    prm = minimize(generate_lms_function(lognorm, median=v_50, p025=v_025, p975=v_975,loc=False),x0,bounds=[(0,100),(v_50-1,v_50+1)]).x
    if rnd_cdf == None:
        rnd = lognorm.rvs(prm[0],0.,prm[1])
        rnd_cdf = lognorm.cdf(rnd,prm[0],0.,prm[1])
    else:
        rnd = lognorm.ppf(rnd_cdf,prm[0],0.,prm[1])
    return rnd,rnd_cdf



def gen_dist_param(mean_all,median_all,v_05_all,v_95_all,v_99_all):
    mean_gen,val_cdf = compute_parameter(v_50=mean_all[0], v_025=mean_all[1], v_975=mean_all[2])
    median_gen,_ = compute_parameter(v_50=median_all[0], v_025=median_all[1], v_975=median_all[2], rnd_cdf=val_cdf)
    v_05_gen,_ = compute_parameter(v_50=v_05_all[0], v_025=v_05_all[1], v_975=v_05_all[2], rnd_cdf=val_cdf)
    v_95_gen,_ = compute_parameter(v_50=v_95_all[0], v_025=v_95_all[1], v_975=v_95_all[2], rnd_cdf=val_cdf)
    v_99_gen,_ = compute_parameter(v_50=v_99_all[0], v_025=v_99_all[1], v_975=v_99_all[2], rnd_cdf=val_cdf)
    
    sigma,mu = minimize(generate_lms_function(lognorm, mean= mean_gen, \
                                              median=median_gen, p05=v_05_gen, p95=v_95_gen, p99=v_99_gen,loc=False),[0.5,median_gen]).x
    loc = 0.
    return sigma,loc,mu