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


folder = "controlled_perturb_all"

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
    print(i)
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
        
Nsim = j

#%% COMPUTE PARAMETERS
active_median = np.median(active[inds],axis=0)
active_01 = np.percentile(active[inds],1,axis=0)
active_05 = np.percentile(active[inds],5,axis=0)
active_10 = np.percentile(active[inds],10,axis=0)
active_25 = np.percentile(active[inds],25,axis=0)
active_75 = np.percentile(active[inds],75,axis=0)
active_90 = np.percentile(active[inds],90,axis=0)
active_95 = np.percentile(active[inds],95,axis=0)
active_99 = np.percentile(active[inds],99,axis=0)

infectious_median = np.median(infectious[inds],axis=0)
infectious_01 = np.percentile(infectious[inds],1,axis=0)
infectious_05 = np.percentile(infectious[inds],5,axis=0)
infectious_10 = np.percentile(infectious[inds],10,axis=0)
infectious_25 = np.percentile(infectious[inds],25,axis=0)
infectious_75 = np.percentile(infectious[inds],75,axis=0)
infectious_90 = np.percentile(infectious[inds],90,axis=0)
infectious_95 = np.percentile(infectious[inds],95,axis=0)
infectious_99 = np.percentile(infectious[inds],99,axis=0)

symptoms_median = np.median(symptoms[inds],axis=0)
symptoms_01 = np.percentile(symptoms[inds],1,axis=0)
symptoms_05 = np.percentile(symptoms[inds],5,axis=0)
symptoms_10 = np.percentile(symptoms[inds],10,axis=0)
symptoms_25 = np.percentile(symptoms[inds],25,axis=0)
symptoms_75 = np.percentile(symptoms[inds],75,axis=0)
symptoms_90 = np.percentile(symptoms[inds],90,axis=0)
symptoms_95 = np.percentile(symptoms[inds],95,axis=0)
symptoms_99 = np.percentile(symptoms[inds],99,axis=0)

hospitalized_median = np.median(hospitalized[inds],axis=0)
hospitalized_01 = np.percentile(hospitalized[inds],1,axis=0)
hospitalized_05 = np.percentile(hospitalized[inds],5,axis=0)
hospitalized_10 = np.percentile(hospitalized[inds],10,axis=0)
hospitalized_25 = np.percentile(hospitalized[inds],25,axis=0)
hospitalized_75 = np.percentile(hospitalized[inds],75,axis=0)
hospitalized_90 = np.percentile(hospitalized[inds],90,axis=0)
hospitalized_95 = np.percentile(hospitalized[inds],95,axis=0)
hospitalized_99 = np.percentile(hospitalized[inds],99,axis=0)

icu_median = np.median(icu[inds],axis=0)
icu_01 = np.percentile(icu[inds],1,axis=0)
icu_05 = np.percentile(icu[inds],5,axis=0)
icu_10 = np.percentile(icu[inds],10,axis=0)
icu_25 = np.percentile(icu[inds],25,axis=0)
icu_75 = np.percentile(icu[inds],75,axis=0)
icu_90 = np.percentile(icu[inds],90,axis=0)
icu_95 = np.percentile(icu[inds],95,axis=0)
icu_99 = np.percentile(icu[inds],99,axis=0)

dead_median = np.median(dead[inds],axis=0)
dead_01 = np.percentile(dead[inds],1,axis=0)
dead_05 = np.percentile(dead[inds],5,axis=0)
dead_10 = np.percentile(dead[inds],10,axis=0)
dead_25 = np.percentile(dead[inds],25,axis=0)
dead_75 = np.percentile(dead[inds],75,axis=0)
dead_90 = np.percentile(dead[inds],90,axis=0)
dead_95 = np.percentile(dead[inds],95,axis=0)
dead_99 = np.percentile(dead[inds],99,axis=0)

immune_median = np.median(immune[inds],axis=0)
immune_01 = np.percentile(immune[inds],1,axis=0)
immune_05 = np.percentile(immune[inds],5,axis=0)
immune_10 = np.percentile(immune[inds],10,axis=0)
immune_25 = np.percentile(immune[inds],25,axis=0)
immune_75 = np.percentile(immune[inds],75,axis=0)
immune_90 = np.percentile(immune[inds],90,axis=0)
immune_95 = np.percentile(immune[inds],95,axis=0)
immune_99 = np.percentile(immune[inds],99,axis=0)

susceptible_median = np.median(susceptible[inds],axis=0)
susceptible_01 = np.percentile(susceptible[inds],1,axis=0)
susceptible_05 = np.percentile(susceptible[inds],5,axis=0)
susceptible_10 = np.percentile(susceptible[inds],10,axis=0)
susceptible_25 = np.percentile(susceptible[inds],25,axis=0)
susceptible_75 = np.percentile(susceptible[inds],75,axis=0)
susceptible_90 = np.percentile(susceptible[inds],90,axis=0)
susceptible_95 = np.percentile(susceptible[inds],95,axis=0)
susceptible_99 = np.percentile(susceptible[inds],99,axis=0)

    
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
        'immune_95': immune_95
        }
df = pd.DataFrame(raw_data, columns = list(raw_data.keys()))
df.to_csv("./"+folder+"slo_pandemic_model_output.csv")


        


#%% PLOT FIELDS
fig = plt.figure(1,figsize=(12,6))

plt.title("COVID-19 Pandemic in Slovenia")

# simulated values
# plt.fill_between(days,active_01,active_99,color="blue",alpha=0.1)
#plt.fill_between(days,active_25,active_75,color="blue",alpha=0.2)
#plt.plot(days,active_median,label="Active",color="blue",lw=3)
plt.fill_between(days,Nall-susceptible_95,Nall-susceptible_05,color="blue",alpha=0.15)
plt.fill_between(days,Nall-susceptible_75,Nall-susceptible_25,color="blue",alpha=0.3)
plt.plot(days,Nall-susceptible_median,label="Infected (cumulative)",color="blue",lw=3)

# plt.fill_between(days,symptoms_01,symptoms_99,color="green",alpha=0.1)
plt.fill_between(days,symptoms_05,symptoms_95,color="lime",alpha=0.15)
plt.fill_between(days,symptoms_25,symptoms_75,color="lime",alpha=0.3)
plt.plot(days,symptoms_median,label="Symptomatic (cumulative)",color="lime",lw=3)

plt.fill_between(days,infectious_05,infectious_95,color="#FF0099",alpha=0.15)
plt.fill_between(days,infectious_25,infectious_75,color="#FF0099",alpha=0.3)
plt.plot(days,infectious_median,label="Infectious",color="#FF0099",lw=3)

# plt.fill_between(days,symptoms_01,symptoms_99,color="green",alpha=0.1)
plt.fill_between(days,hospitalized_05,hospitalized_95,color="orange",alpha=0.15)
plt.fill_between(days,hospitalized_25,hospitalized_75,color="orange",alpha=0.3)
plt.plot(days,hospitalized_median,label="Hospitalised",color="orange",lw=3)

plt.fill_between(days,icu_05,icu_95,color="brown",alpha=0.15)
plt.fill_between(days,icu_25,icu_75,color="brown",alpha=0.3)
plt.plot(days,icu_median,label="ICU",color="brown",lw=3)

plt.fill_between(days,dead_05,dead_95,color="black",alpha=0.15,label="90 %") 
plt.fill_between(days,dead_25,dead_75,color="black",alpha=0.3,label="50 %")
plt.plot(days,dead_median,label="Dead (cumulative)",color="black",lw=3)

#plt.fill_between(days[1:],dead_05[1:]-dead_05[:-1],dead_95[1:]-dead_95[:-1],color="black",alpha=0.15,label="90 %") 
#plt.fill_between(days[1:],dead_25[1:]-dead_25[:-1],dead_75[1:]-dead_75[:-1],color="black",alpha=0.3,label="50 %")
#plt.plot(days[1:],dead_median[1:]-dead_median[:-1],label="Dead (cumulative)",color="black",lw=3)

# plt.plot(days[:20],contagious[:20]/7.5,'b--',label="Active cases")
# plt.plot(days,immune,label="Immune",color="green")
# plt.plot(days,critical,label="Critical (ICU)",color="red",lw=2)
# plt.plot(days,dead,'k-',label="Dead")
# plt.plot(days,[res_num]*len(days),'r--',label="Future healthcare capacity (125 respirators)")


# real data
plt.plot(data_days,data_cases,'go',label="Positive cases (data)",markeredgecolor="k",markeredgewidth=0.4,markersize=4)
plt.plot(data_days,data_hosp,'o',color='orange',label="Hospitalized (data)",markeredgecolor="k",markeredgewidth=0.4,markersize=4)
plt.plot(data_days,data_icu,'o',color='brown',label="ICU (data)",markeredgecolor="k",markeredgewidth=0.4,markersize=4)
plt.plot(data_days,data_dead,'ko',label="Dead (data)",markeredgecolor="white",markeredgewidth=0.4,markersize=4)

#plt.plot(data_days[1:],data_dead[1:]-data_dead[:-1],'ko',label="Dead (data)",markeredgecolor="white",markeredgewidth=0.4,markersize=4)


plt.yscale('log')

xticks_lbls = []
date0 = datetime.datetime(2020,3,12)
for i in range(Nt):
    date = date0+datetime.timedelta(i)
    xticks_lbls.append(date.strftime("%B %d"))
plt.xticks(range(0,Nt,10),xticks_lbls[::10],rotation=40)
# plt.fill_between(days, res_num ,0,alpha=0.2,color='r')
# plt.fill_between(days,critical,res_num)
plt.ylabel("Number of nodes")
plt.grid(b=True, which='major', color='grey', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.ylim([1,10**5])
plt.xlim([-1,Nt])
plt.legend()
fig.savefig("potek_pandemije{:02d}.png".format(run),dpi=250)
   
