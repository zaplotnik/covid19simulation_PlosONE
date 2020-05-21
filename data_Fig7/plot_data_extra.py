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

N = 5
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

Ncons = 100
day_infected_res = np.zeros(Ncons)
day_infected_count  = np.zeros(Ncons)

r0_res = np.zeros((N,Nt))
r0_house_res = np.zeros((N,Nt))
r0_other_res = np.zeros((N,Nt))

j = 0
for i in range(N):
    print(i)
    if os.path.exists("./model_output7/tab_active_{:03d}.txt".format(i)):
        status_r0 = np.load("./model_output7/r0_{:03d}.npy".format(i))
        status_r0_other = np.load("./model_output7/r0_other_{:03d}.npy".format(i))
        status_r0_house = np.load("./model_output7/r0_house_{:03d}.npy".format(i))
        day_infected = np.load("./model_output7/day_infected_{:03d}.npy".format(i))
        rands_input = np.load("./model_output7/rands_input_{:03d}.npy".format(i))
        
        active[j,:] = np.loadtxt("./model_output7/tab_active_{:03d}.txt".format(i))
        infectious[j,:] = np.loadtxt("./model_output7/tab_infectious_{:03d}.txt".format(i))
        incubation[j,:] = np.loadtxt("./model_output7/tab_incubation_{:03d}.txt".format(i))
        symptoms[j,:] = np.loadtxt("./model_output7/tab_symptoms_{:03d}.txt".format(i))
        hospitalized[j,:] = np.loadtxt("./model_output7/tab_hospitalized_{:03d}.txt".format(i))
        icu[j,:] = np.loadtxt("./model_output7/tab_icu_{:03d}.txt".format(i))
        dead[j,:] = np.loadtxt("./model_output7/tab_dead_{:03d}.txt".format(i))
        immune[j,:] = np.loadtxt("./model_output7/tab_immune_{:03d}.txt".format(i))
        susceptible[j,:] = np.loadtxt("./model_output7/tab_susceptible_{:03d}.txt".format(i))
        
        L_day_infected = (day_infected>0)
        for ncon in range(Ncons):
            inds = (np.where((rands_input > ncon) & (rands_input < ncon+1) & L_day_infected))[0]
            if inds.size == 0:
                add = 0
            else:
                add = day_infected[inds].sum()
                day_infected_count[ncon] += inds.size
            #print ncon," < #con < ",ncon+1, ": ", add, inds.shape[0]
            day_infected_res[ncon] += add        
        
        inc_r0_inf = (np.where(status_r0>-1))[0]
        inc_r0_inf_house = (np.where(status_r0_house>-1))[0]
        inc_r0_inf_other = (np.where(status_r0_other>-1))[0]
        
        tabt_r0 = day_infected[inc_r0_inf] 
        #print tabt_r0,(status_r0[inc_r0_inf]>-1).sum()
        tabt_r0_house = day_infected[inc_r0_inf_house]
        tabt_r0_other = day_infected[inc_r0_inf_other]
        # day_sort = np.argsort(tab1)
        # tab_day = day_infected[day_sort]
        
        blah_r0 = (status_r0[inc_r0_inf])
        blah_r0_house = (status_r0_house[inc_r0_inf_house])
        blah_r0_other = (status_r0_other[inc_r0_inf_other])
        #print blah_r0,blah_r0_house,blah_r0_other
        tab_r0 = np.zeros(Nt)
        tab_r0_house = np.zeros(Nt)
        tab_r0_other = np.zeros(Nt)
        for t in range(Nt):
            incs = np.where(tabt_r0 == t)[0]
            if incs.size == 0:
                tab_r0[t] = 0.
            else:
                tab_r0[t] = blah_r0[incs].mean()

            incs = np.where(tabt_r0_house == t)[0]
            if incs.size == 0:
                tab_r0_house[t] = 0.
            else:
                tab_r0_house[t] = (blah_r0_house[incs]).mean()
            
            incs = np.where(tabt_r0_other == t)[0]
            if incs.size == 0:
                tab_r0_other[t] = 0.
            else:
                tab_r0_other[t] = (blah_r0_other[incs]).mean()
        
        #print tab_r0,tab_r0_house,tab_r0_other
         
        r0_res[j,:] = tab_r0
        r0_house_res[j,:] = tab_r0_house
        r0_other_res[j,:] = tab_r0_other

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

day_infected_res = day_infected_res/day_infected_count


r0_median = np.median(r0_res[inds],axis=0)
r0_05= np.percentile(r0_res[inds],5,axis=0)
r0_25= np.percentile(r0_res[inds],25,axis=0)
r0_75= np.percentile(r0_res[inds],75,axis=0)
r0_95 = np.percentile(r0_res[inds],95,axis=0)

r0_house_median = np.median(r0_house_res[inds],axis=0)
r0_house_05= np.percentile(r0_house_res[inds],5,axis=0)
r0_house_25= np.percentile(r0_house_res[inds],25,axis=0)
r0_house_75= np.percentile(r0_house_res[inds],75,axis=0)
r0_house_95 = np.percentile(r0_house_res[inds],95,axis=0)

r0_other_median = np.median(r0_other_res[inds],axis=0)
r0_other_05= np.percentile(r0_other_res[inds],5,axis=0)
r0_other_25= np.percentile(r0_other_res[inds],25,axis=0)
r0_other_75= np.percentile(r0_other_res[inds],75,axis=0)
r0_other_95 = np.percentile(r0_other_res[inds],95,axis=0)

#np.save("r0_median",r0_median)
#np.save("r0_25",r0_25)
#np.save("r0_75",r0_75)

#np.save("r0_house_median",r0_house_median)
#np.save("r0_house_25",r0_house_25)
#np.save("r0_house_75",r0_house_75)

#np.save("r0_other_median",r0_other_median)
#np.save("r0_other_25",r0_other_25)
#np.save("r0_other_75",r0_other_75)

r0_median = np.load("r0_median.npy")
r0_25 = np.load("r0_25.npy")
r0_75 = np.load("r0_75.npy")

r0_house_median = np.load("r0_house_median.npy")
r0_house_25 = np.load("r0_house_25.npy")
r0_house_75 = np.load("r0_house_75.npy")

r0_other_median = np.load("r0_other_median.npy")
r0_other_25 = np.load("r0_other_25.npy")
r0_other_75 = np.load("r0_other_75.npy")

xticks_lbls = []
date0 = datetime.datetime(2020,3,12)
for i in range(Nt):
    date = date0+datetime.timedelta(i)
    xticks_lbls.append(date.strftime("%B %d"))  

fig2= plt.figure(2,figsize=(6,4))
plt.ylabel(r"Reproduction number $R_0$")
#plt.fill_between(days+4.,r0_05,r0_95,color="blue",alpha=0.15)
plt.fill_between(days+5.,r0_25,r0_75,color="k",alpha=0.3)
plt.plot(days+5.,r0_median,color="k",lw=3,label="all")
plt.fill_between(days+5.,r0_house_25,r0_house_75,color="blue",alpha=0.3)
plt.plot(days+5.,r0_house_median,color="blue",lw=3,label="household")
plt.fill_between(days+5.,r0_other_25,r0_other_75,color="red",alpha=0.3)
plt.plot(days+5.,r0_other_median,color="red",lw=3,label="outer")
plt.xticks(range(0,Nt,10),xticks_lbls[::10],rotation=40)
plt.xlim([-1,Nt-20])
plt.ylim([0,1.5])
plt.grid()
plt.legend()
plt.tight_layout()
fig2.savefig("reproduction_number{:02d}.png".format(run),dpi=300)

