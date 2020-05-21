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
import sys


folder = sys.argv[1]#"controlled_perturb_transmission"

#%% READ SIMULATED DATA
days = np.loadtxt("./"+folder+"/model_output/tab_days_001.txt")
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

j = 0
for i in range(N):
    
    if os.path.exists("./"+folder+"/model_output/tab_active_{:03d}.txt".format(i)):

        active[j,:] = np.loadtxt("./"+folder+"/model_output/tab_active_{:03d}.txt".format(i))
        infectious[j,:] = np.loadtxt("./"+folder+"/model_output/tab_infectious_{:03d}.txt".format(i))
        incubation[j,:] = np.loadtxt("./"+folder+"/model_output/tab_incubation_{:03d}.txt".format(i))
        symptoms[j,:] = np.loadtxt("./"+folder+"/model_output/tab_symptoms_{:03d}.txt".format(i))
        hospitalized[j,:] = np.loadtxt("./"+folder+"/model_output/tab_hospitalized_{:03d}.txt".format(i))
        icu[j,:] = np.loadtxt("./"+folder+"/model_output/tab_icu_{:03d}.txt".format(i))
        dead[j,:] = np.loadtxt("./"+folder+"/model_output/tab_dead_{:03d}.txt".format(i))
        immune[j,:] = np.loadtxt("./"+folder+"/model_output/tab_immune_{:03d}.txt".format(i))
        susceptible[j,:] = np.loadtxt("./"+folder+"/model_output/tab_susceptible_{:03d}.txt".format(i))
        j = j +1

Nsim = j

#%% COMPUTE PARAMETERS
active_median = np.median(active,axis=0)
active_01 = np.percentile(active,1,axis=0)
active_05 = np.percentile(active,5,axis=0)
active_10 = np.percentile(active,10,axis=0)
active_25 = np.percentile(active,25,axis=0)
active_75 = np.percentile(active,75,axis=0)
active_90 = np.percentile(active,90,axis=0)
active_95 = np.percentile(active,95,axis=0)
active_99 = np.percentile(active,99,axis=0)

infectious_median = np.median(infectious,axis=0)
infectious_01 = np.percentile(infectious,1,axis=0)
infectious_05 = np.percentile(infectious,5,axis=0)
infectious_10 = np.percentile(infectious,10,axis=0)
infectious_25 = np.percentile(infectious,25,axis=0)
infectious_75 = np.percentile(infectious,75,axis=0)
infectious_90 = np.percentile(infectious,90,axis=0)
infectious_95 = np.percentile(infectious,95,axis=0)
infectious_99 = np.percentile(infectious,99,axis=0)

symptoms_median = np.median(symptoms,axis=0)
symptoms_01 = np.percentile(symptoms,1,axis=0)
symptoms_05 = np.percentile(symptoms,5,axis=0)
symptoms_10 = np.percentile(symptoms,10,axis=0)
symptoms_25 = np.percentile(symptoms,25,axis=0)
symptoms_75 = np.percentile(symptoms,75,axis=0)
symptoms_90 = np.percentile(symptoms,90,axis=0)
symptoms_95 = np.percentile(symptoms,95,axis=0)
symptoms_99 = np.percentile(symptoms,99,axis=0)

hospitalized_median = np.median(hospitalized,axis=0)
hospitalized_01 = np.percentile(hospitalized,1,axis=0)
hospitalized_05 = np.percentile(hospitalized,5,axis=0)
hospitalized_10 = np.percentile(hospitalized,10,axis=0)
hospitalized_25 = np.percentile(hospitalized,25,axis=0)
hospitalized_75 = np.percentile(hospitalized,75,axis=0)
hospitalized_90 = np.percentile(hospitalized,90,axis=0)
hospitalized_95 = np.percentile(hospitalized,95,axis=0)
hospitalized_99 = np.percentile(hospitalized,99,axis=0)

icu_median = np.median(icu,axis=0)
icu_01 = np.percentile(icu,1,axis=0)
icu_05 = np.percentile(icu,5,axis=0)
icu_10 = np.percentile(icu,10,axis=0)
icu_25 = np.percentile(icu,25,axis=0)
icu_75 = np.percentile(icu,75,axis=0)
icu_90 = np.percentile(icu,90,axis=0)
icu_95 = np.percentile(icu,95,axis=0)
icu_99 = np.percentile(icu,99,axis=0)

dead_median = np.median(dead,axis=0)
dead_01 = np.percentile(dead,1,axis=0)
dead_05 = np.percentile(dead,5,axis=0)
dead_10 = np.percentile(dead,10,axis=0)
dead_25 = np.percentile(dead,25,axis=0)
dead_75 = np.percentile(dead,75,axis=0)
dead_90 = np.percentile(dead,90,axis=0)
dead_95 = np.percentile(dead,95,axis=0)
dead_99 = np.percentile(dead,99,axis=0)

immune_median = np.median(immune,axis=0)
immune_01 = np.percentile(immune,1,axis=0)
immune_05 = np.percentile(immune,5,axis=0)
immune_10 = np.percentile(immune,10,axis=0)
immune_25 = np.percentile(immune,25,axis=0)
immune_75 = np.percentile(immune,75,axis=0)
immune_90 = np.percentile(immune,90,axis=0)
immune_95 = np.percentile(immune,95,axis=0)
immune_99 = np.percentile(immune,99,axis=0)

susceptible_median = np.median(susceptible,axis=0)
susceptible_01 = np.percentile(susceptible,1,axis=0)
susceptible_05 = np.percentile(susceptible,5,axis=0)
susceptible_10 = np.percentile(susceptible,10,axis=0)
susceptible_25 = np.percentile(susceptible,25,axis=0)
susceptible_75 = np.percentile(susceptible,75,axis=0)
susceptible_90 = np.percentile(susceptible,90,axis=0)
susceptible_95 = np.percentile(susceptible,95,axis=0)
susceptible_99 = np.percentile(susceptible,99,axis=0)

    
dates = [(datetime.datetime(2020,3,12) + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(0,Nt)]
  
raw_data = {
        'date' : dates,
        'active_median': active_median, 
        'active_05': active_05,
        'active_25': active_25, 
        'active_75': active_75,
        'active_95': active_95,
        'infectious_median': infectious_median, 
        'infectious_05': infectious_05,
        'infectious_25': infectious_25, 
        'infectious_75': infectious_75,
        'infectious_95': infectious_95, 
        'symptoms_median': symptoms_median, 
        'symptoms_05': symptoms_05,
        'symptoms_25': symptoms_25, 
        'symptoms_75': symptoms_75,
        'symptoms_95': symptoms_95, 
        'hospitalized_median': hospitalized_median, 
        'hospitalized_05': hospitalized_05,
        'hospitalized_25': hospitalized_25, 
        'hospitalized_75': hospitalized_75,
        'hospitalized_95': hospitalized_95, 
        'icu_median': icu_median, 
        'icu_05': icu_05,
        'icu_25': icu_25, 
        'icu_75': icu_75,
        'icu_95': icu_95, 
        'dead_median': dead_median, 
        'dead_05': dead_05,
        'dead_25': dead_25, 
        'dead_75': dead_75,
        'dead_95': dead_95,
        'immune_median': immune_median, 
        'immune_05': immune_05,
        'immune_25': immune_25, 
        'immune_75': immune_75,
        'immune_95': immune_95,
        'susceptible_median': susceptible_median, 
        'susceptible_05': susceptible_05,
        'susceptible_25': susceptible_25, 
        'susceptible_75': susceptible_75,
        'susceptible_95': susceptible_95
        }
df = pd.DataFrame(raw_data, columns = list(raw_data.keys()))
df.to_csv("./"+folder+".csv")
   
