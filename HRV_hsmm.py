# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:07:02 2022

@author: nct76
"""

import pyhsmm
import pyhsmm.basic.distributions as distributions
import numpy as np
from matplotlib import pyplot as plt


import numpy as np
import pyedflib
file_name = 'C:/Users/nct76/Downloads/ForasData/09-20-57.EDF'
f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
signal_labels = f.getSignalLabels()

#%%
ecg = f.readSignal(0)
accX = f.readSignal(1)
accY = f.readSignal(2)
accZ = f.readSignal(3)
hrv = f.readSignal(5)
#%%
ecg_sf = f.getSampleFrequencies()[0]
accX_sf = f.getSampleFrequencies()[1]
accY_sf = f.getSampleFrequencies()[2]
accZ_sf = f.getSampleFrequencies()[3]
hrv_sf = f.getSampleFrequencies()[5]

startdate = f.getStartdatetime()

ecg_t = np.arange(1,len(ecg)+1)/ecg_sf
accX_t = np.arange(1,len(accX)+1)/accX_sf
accY_t = np.arange(1,len(accY)+1)/accY_sf
accZ_t = np.arange(1,len(accZ)+1)/accZ_sf
hrv_t = np.arange(1,len(hrv)+1)/hrv_sf

#%%
hrv_t_ms = hrv_t*1000
hrv_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + hrv_t_ms.astype(int).astype('timedelta64[ms]')

accX_t_ms = accX_t*1000
accX_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accX_t_ms.astype(int).astype('timedelta64[ms]')

accY_t_ms = accY_t*1000
accY_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accY_t_ms.astype(int).astype('timedelta64[ms]')

accZ_t_ms = accZ_t*1000
accZ_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + accZ_t_ms.astype(int).astype('timedelta64[ms]')

ecg_t_ms = ecg_t*1000
ecg_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + ecg_t_ms.astype(int).astype('timedelta64[ms]')

#%%import more_itertools as mit
import more_itertools as mit

hrv_calc_len = 5 #minutes
hrv_chunkedlen = hrv_calc_len * 300

rrs = [np.array(x)[(np.array(x)>0) & (np.array(x)<1200)] for x in mit.chunked(hrv, hrv_chunkedlen)]
#%%

rdff = np.array([np.sqrt(sum((r[:-1] - r[1:])**2)/(len(r)-1)) if len(r)>200 else 0 for r in rrs])
hr = np.array([60*1000*(1/np.mean(r)) if len(r)>200 else 0 for r in rrs])
rdff_t = np.arange(0,len(rdff))*(60 *hrv_calc_len)
rdff_t_ms = rdff_t*1000

rdff_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + rdff_t_ms.astype(int).astype('timedelta64[ms]')
#%%
plt.plot(rdff_t_ms, rdff)
plt.plot(rdff_t_ms, hr)

plt.plot(accZ_t_ms, accZ/100)
#%%
from scipy import signal

acc = np.array([accX,accY,accZ]).T
enmo = np.maximum(np.sqrt(np.sum(np.power(np.array(acc),2),axis=1))-1000,0)
samp = round(len(enmo)/len(rdff))
enmo_ds = [np.mean(x) for x in mit.chunked(enmo, samp)]
data_t = np.arange(0,len(enmo_ds))*(60 *samp)
data_t_ms = data_t*1000
# nonphys = rdff>200 
# hr[rdff>200] = np.mean(hr[rdff<=200])
# rdff[rdff>200] = np.mean(rdff[rdff<=200])
data_t_ms =np.datetime64(startdate).astype('datetime64[ms]') + rdff_t_ms.astype(int).astype('timedelta64[ms]')

#%%
data = np.array([enmo_ds, hr, rdff]).T

t_ms = rdff_t_ms[np.isnan(data[:,2]) != True]
data = data[np.isnan(data[:,2]) != True]

plt.plot(data[:,0],data[:,1],'kx')
plt.xlabel('Acceleration (g)')
plt.ylabel('HRV (ms)')
from pyhsmm.util.plot import pca_project_data
plt.figure()
plt.plot(pca_project_data(data,1))

#%%
import pyhsmm
import pyhsmm.basic.distributions as distributions
import copy
obs_dim = 3
Nmax = 25

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+5}
dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6., # better to sample over these; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)

posteriormodel.add_data(data,trunc=60)

models = []
for idx in range(0,500):
    posteriormodel.resample_model()

        
# fig = plt.figure()
# for idx, model in enumerate(models):
#     plt.clf()
#     model.plot()
#     plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (10*(idx+1)))
#     plt.savefig('iter_%.3d.png' % (10*(idx+1)))

#%%
import scipy.stats as sst
import scipy.signal as sgl
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

state_mean_amplitudes = []
state_mean_hrv = []

state_mean_hr = []
state_mean_durations = []
state_numbers = []
states_used = np.squeeze(np.where(posteriormodel.state_usages>0))
n = 0
for i in states_used:
    state_mean_amplitudes.append(posteriormodel.obs_distns[i].mu[0])
    state_mean_hr.append(posteriormodel.obs_distns[i].mu[1])
    state_mean_hrv.append(posteriormodel.obs_distns[i].mu[2])
    state_mean_durations.append(posteriormodel.dur_distns[i].mean)
#%%
state_mean_amplitudes = np.array(state_mean_amplitudes)
state_mean_hr = np.array(state_mean_hr)
state_mean_hrv = np.array(state_mean_hrv)
usage = posteriormodel.state_usages[posteriormodel.state_usages>0]

plt.scatter(state_mean_hr[state_mean_hr<200], state_mean_amplitudes[state_mean_hr<200],
            usage[state_mean_hr<200]*1000, c=state_mean_hrv[state_mean_hr<200])
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Acc (g)')
cbar = plt.colorbar()
cbar.set_label('HRV (ms)')
ss = np.array(posteriormodel.stateseqs[n]).astype(float)
ss_int = np.array(posteriormodel.stateseqs[n]).astype(int)
orderofamp = np.argsort(state_mean_amplitudes)
amprank= sst.rankdata(state_mean_amplitudes)
amprank = amprank.astype(int)

hr_ss = np.array(posteriormodel.stateseqs[n]).astype(float)
hr_ss_int = np.array(posteriormodel.stateseqs[n]).astype(int)
hr_orderofamp = np.argsort(state_mean_hr)
hr_amprank= sst.rankdata(state_mean_hr)
hr_amprank = hr_amprank.astype(int)

hrv_ss = np.array(posteriormodel.stateseqs[n]).astype(float)
hrv_ss_int = np.array(posteriormodel.stateseqs[n]).astype(int)
hrv_orderofamp = np.argsort(state_mean_hrv)
hrv_amprank= sst.rankdata(state_mean_hrv)
hrv_amprank = hrv_amprank.astype(int)

statesnum = ss
state_list = []
for s_i in ss_int:
    state_list.append(np.squeeze(np.where(states_used==s_i)))
state_list = np.array(state_list)
#%%
ss_int = amprank[state_list]
hrv_ss_int = amprank[state_list]
hrv_ss_int = amprank[state_list]
for i in  posteriormodel.used_states:
    ss[ss==i] = posteriormodel.obs_distns[i].mu[0]
    hr_ss[hr_ss==i] = posteriormodel.obs_distns[i].mu[1]
    hrv_ss[hrv_ss==i] = posteriormodel.obs_distns[i].mu[2]
    
from matplotlib.cm import ScalarMappable
from pylab import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

cmap_name = 'my_list'
cmap = cm.get_cmap('Set3', 8)

color = 'k'
fig, ax1 = plt.subplots()
# ax1.set_xlabel('time (minutes)',fontsize=20)
# ax1.set_ylabel('Acceleration (mg)', color=color,fontsize=20)
ax1.plot(t_ms,posteriormodel.datas[n],color=color,linewidth=0.8)


color = 'tab:red'
cc=[[0,0,0]]*len(ss)
ax1.plot(t_ms,ss,'k',label='Acc',linewidth=0.8)
ax1.plot(t_ms,hrv_ss,'b',label='HRV',linewidth=0.8)
ax1.plot(t_ms,hr_ss,'g',label='HR',linewidth=0.8)

bb = ax1.bar(t_ms,150,bottom=0, width=0.1, color=cmap(ss_int-1), align='edge')
#bb = ax1.bar(t_ms,nonphys*150,bottom=0, width=0.1, color='k', align='edge')
plt.legend()