#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 22:14:39 2020

@author: ziga
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
from scipy.stats import *

run = 5



save_data = True
#%% READ REAL DATA
# https://github.com/slo-covid-19/data/blob/master/csv/stats.csv
data_stats = pd.read_csv(r"https://raw.githubusercontent.com/slo-covid-19/data/master/csv/stats.csv",\
                         index_col="date", usecols=["date", "tests.positive.todate","state.in_hospital", "state.icu","state.deceased.todate"], parse_dates=["date"])


# day 0 = March 12
data_hospitalised = data_stats["state.in_hospital"][17:]
data_icu = data_stats["state.icu"][17:]
data_cases = data_stats["tests.positive.todate"][17:]
data_dead = data_stats["state.deceased.todate"][17:]
#data_cases_new = data_stats["tests.positive"][17:]
data_days = np.arange(0,data_hospitalised.shape[0])
Ndata = len(data_dead)


#%% READ SIMULATED DATA
days = np.loadtxt("./model_output7/tab_days_001.txt")
Nt =days.shape[0]

N = 500
active = np.zeros((N,days.shape[0]))
infectious = np.zeros((N,days.shape[0]))
incubation = np.zeros((N,days.shape[0]))
symptoms = np.zeros((N,days.shape[0]))
hospitalized = np.zeros((N,days.shape[0]))
icu = np.zeros((N,days.shape[0]))
dead = np.zeros((N,days.shape[0]))
immune = np.zeros((N,days.shape[0]))
susceptible = np.zeros((N,days.shape[0]))

Nall = int(2.045795*10**6)

inc_dead_data = data_dead.values>0
inc_icu_data = data_icu.values>0

blah = np.zeros((2,N))

tab_params = np.zeros((N,32))

j = 0
for i in range(N):
    print(i)
    if os.path.exists("./model_output7/tab_active_{:03d}.txt".format(i)):
        
        active[j,:] = np.loadtxt("./model_output7/tab_active_{:03d}.txt".format(i))
        infectious[j,:] = np.loadtxt("./model_output7/tab_infectious_{:03d}.txt".format(i))
        incubation[j,:] = np.loadtxt("./model_output7/tab_incubation_{:03d}.txt".format(i))
        symptoms[j,:] = np.loadtxt("./model_output7/tab_symptoms_{:03d}.txt".format(i))
        hospitalized[j,:] = np.loadtxt("./model_output7/tab_hospitalized_{:03d}.txt".format(i))
        icu[j,:] = np.loadtxt("./model_output7/tab_icu_{:03d}.txt".format(i))
        dead[j,:] = np.loadtxt("./model_output7/tab_dead_{:03d}.txt".format(i))
        immune[j,:] = np.loadtxt("./model_output7/tab_immune_{:03d}.txt".format(i))
        susceptible[j,:] = np.loadtxt("./model_output7/tab_susceptible_{:03d}.txt".format(i))
        
        tab_params[j,:]  = np.loadtxt("./model_output7/params_{:03d}.txt".format(i))


        dead_model = dead[j,:Ndata] > 0
        icu_model = icu[j,:Ndata] > 0       

        inc_dead_all = inc_dead_data & dead_model
        inc_icu_all = inc_icu_data & icu_model
        
        #print dead[j,:Ndata][inc_dead_all]
        #print (data_dead.values)[inc_dead_all]   

        #print icu[j,:Ndata][inc_icu_all]
        #print (data_icu.values)[inc_icu_all]      

        sum1 = (np.abs(np.log((data_dead.values)[inc_dead_all]) - np.log(dead[j,:Ndata][inc_dead_all]))).sum()
        sum2 = (np.abs((np.log(data_icu.values)[inc_icu_all]) - np.log(icu[j,:Ndata][inc_icu_all]))).sum()       
        print sum1, sum2
        #print ""
        blah[0,j] = i
        blah[1,j] = sum1 + sum2 
        j += 1
    else:
        continue

Nsim = j
print Nsim
a =  np.percentile(blah[1,:Nsim],10)
print a
inds = blah[0,:j][blah[1,:Nsim] < a]
inds = inds.astype(np.int)

print inds

tab_params_select = tab_params[inds]

np.save("tab_params_all",tab_params)
np.save("tab_params_select",tab_params_select)



