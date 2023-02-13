#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:53:29 2022

@author: yanlan
"""

import numpy as np
import pandas as pd
import glob
import imageio
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])
#arrayid = 40 # 40

def get_qc_pixel(qcpath,i,j,r,c,z):
    QC = np.zeros([r,c,z])
    flist = glob.glob(qcpath+str(i)+str(j)+'*QA*.mp4')
    flist.sort()
    print(flist)
    if len(flist)==1:
        nid = imageio.get_reader(flist[0],'ffmpeg')
        for i_time in range(z):
            QC[:,:,i_time] = np.sum(nid.get_data(i_time)[:]>128,axis=2) # cloud, snow, water
        nid.close()

    else: # several tiles were separated into two parts due to limitation of maximum number of exportable pixels on GEE
        nid = imageio.get_reader(flist[0],'ffmpeg')
        for i_time in range(z):
            try:
                img = nid.get_data(i_time)[:]
            except IndexError:
                t0 = np.copy(i_time)
                nid.close()
                break
            QC[:,:,i_time] = np.sum(img>128,axis=2) # cloud, snow, water

        nid = imageio.get_reader(flist[1],'ffmpeg')
        for i_time in range(t0,z):
            img = nid.get_data(i_time-t0)[:]
            QC[:,:,i_time] = np.sum(img>128,axis=2) # cloud, snow, water
    return QC

def get_3d_sr(srpath,Btag,i_band,i,j):
    ''' outputs the surface reflectance array for one band, one tile'''
    print('btag', Btag) #btag should be "b3", 'B6', 'B7'
    print('fname', srpath+str(i)+str(j)+Btag)
    fname = glob.glob(srpath+str(i)+str(j)+Btag+'*.mp4')[0]
    nid = imageio.get_reader(fname,'ffmpeg')
    img0 = nid.get_data(0)[:,:,0]
    img_csv = pd.read_csv(srpath+'TimeStamp_QualityControl/' + str(i) + str(j) + 'ImgQuality.csv')

    date = pd.Series(['-'.join([itm[-8:-4],itm[-4:-2],itm[-2:]]) for itm in img_csv['system:index']])
    sr_3d = np.zeros([img0.shape[0],img0.shape[1],len(img_csv)])

    for i_time in range(len(img_csv)):
        sr_3d[:,:,i_time] = np.array(nid.get_data(i_time)[:,:,i_band],dtype=float)/255
    nid.close()

    if i_band==0 and Btag == band_tags[0]:
        QC = get_qc_pixel(srpath+'QC_PIXEL/',i,j,img0.shape[0],img0.shape[1],len(img_csv))

    else:
        QC = 0
    print(QC)
    return sr_3d, date, QC


def Harmanization(x,sensor, band_tag, i_band):
    if sensor==8:
        if band_tag==0: #B321
            if i_band==0: # red
                xh = 0.0029 + 1.0729*(x)-0.4268*(x**2)
            elif i_band==1: # green
                xh = 0.0092 + 0.7465*(x)+3.0334*(x**2)-10.6023*(x**3)
            elif i_band==2: # blue
                xh = 0.0119 + 0.7782*(x)+4.0941*(x**2)-22.12*(x**3)
        elif band_tag==1: #'B754': 
            if i_band==0: #swir2
                xh = 0.0027 + 0.9058*(x) + 0.589*(x**2)-0.9552*(x**3)
            elif i_band==1: # swir1
                xh = 0.0059 + 0.9681*(x) + 0.547*(x**2)
            elif i_band==2: #nir
                xh = 0.0167 + 0.9068*(x) - 0.0992*(x**2) + 0.1113*(x**3)
        elif band_tag==2: #'B6':
            if i_band==0: # sr.b6
                xh = x.copy()
    elif sensor==7:
        xh = x.copy()
    elif sensor==5:
        if band_tag==0:#B321':
            if i_band==0: # red
                xh = -0.0035 + 0.8783*(x) + 1.5849*(x**2) - 5.8961*(x**3)
            elif i_band==1: # green
                xh = 0.0048 + 0.6984*(x) + 2.0075*(x**2) - 3.9285*(x**3)
            elif i_band==2: # blue
                xh = 0.0105 + 0.2583*(x) + 10.7637*(x**2) - 48.0339*(x**3)
        elif band_tag == 1:#'B754': 
            if i_band==0: #swir2
                xh = 0.0027 + 0.9151*(x) + 0.5465*(x**2) - 1.0338*(x**3)
            elif i_band==1: # swir1
                xh = 0.0024 + 0.9593*(x) + 0.066*(x**2)
            elif i_band==2: #nir
                xh = 0.0108 + 0.9226*(x) + 0.1028*(x**2)
        elif band_tag==2:#'B6':
            if i_band==0: # sr.b6
                xh = x.copy()
    return xh

datapath = '/fs/scratch/PAS2094/ARF/'
band_tags = ['B321', 'B754', 'B6']
unique_year = np.arange(2005,2021)

sensor_list = [7]
outpath = datapath+'L7_Tiles/'
#sensor_list = [5,7,8]
#outpath = datapath+'Harmonized_Tiles/'

i = arrayid//6
j = arrayid-i*6
SR_harmonized_allbands = []
for band_tag in range(len(band_tags)):
    print(band_tag)
    for i_band in range(len(band_tags[band_tag])-1):
        SR_band1_annual_sensors = []
        for i_sensor in sensor_list:
            SR_band1, date, qc = get_3d_sr(datapath+'SurfaceReflectance_L'+str(i_sensor)+'/',band_tags[band_tag],i_band,i,j)

            if band_tag==0 and i_band==0: 
                QC = np.copy(qc); print(np.unique(QC))
                #print(SR_band1.shape, QC.shape)
                #plt.figure();plt.imshow(SR_band1[:,:,0]);plt.colorbar();plt.title('band1, before QC'); plt.savefig('band1_beforeQC.png',dpi=300,bbox_inches='tight')
                #plt.figure();plt.imshow(QC[:,:,0]);plt.colorbar();plt.title('QC'); plt.savefig('QC.png',dpi=300,bbox_inches='tight')
            SR_band1[QC>0] = np.nan
            SR_band1[SR_band1==0] = np.nan
#            if band_tag==0 and i_band==0:
#                plt.figure();plt.imshow(SR_band1[:,:,0]);plt.colorbar();plt.title('band1, after QC'); plt.savefig('band1_afterQC.png',dpi=300,bbox_inches='tight')

            month = np.array([int(itm[5:7]) for itm in date])
            year = np.array([int(itm[:4]) for itm in date])

            SR_band1_annual = np.zeros([SR_band1.shape[0],SR_band1.shape[1],len(unique_year)])+np.nan
            for y in unique_year:
                filter_year =  (month>=6) & (month<=8) & (year==y)
                if sum(filter_year)>0:
                    SR_band1_annual[:,:,y-min(unique_year)] = np.nanmean(SR_band1[:,:,filter_year],axis=2)
            SR_band1_annual = Harmanization(SR_band1_annual,i_sensor,band_tag,i_band)
            SR_band1_annual_sensors.append(SR_band1_annual)
            
        SR_harmonized = np.zeros([SR_band1.shape[0],SR_band1.shape[1],len(unique_year),len(sensor_list)])+np.nan
        for ii in range(len(sensor_list)):
            SR_harmonized[:,:,:,ii] = SR_band1_annual_sensors[ii].copy()
        SR_harmonized = np.nanmean(SR_harmonized,axis=3) # averaged over sensors, annual mean
        SR_harmonized_allbands.append(SR_harmonized) # including all bands
np.save(outpath+'SR_harmonized_'+str(i)+str(j)+'.npy',SR_harmonized_allbands)


