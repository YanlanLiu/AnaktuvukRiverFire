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
#arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])
arrayid = 1 # 40

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

    QC = get_qc_pixel(srpath+'QC_PIXEL/',i,j,img0.shape[0],img0.shape[1],len(img_csv))
    return sr_3d, date, QC


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
    for i_band in range(1):#nange(len(band_tags[band_tag])-1):
        # SR_band1_all_sensors = np.array([]) # all sensors, all images
        month = []
        year = []
        
        for i_sensor in sensor_list:
            SR_band1, date, QC = get_3d_sr(datapath+'SurfaceReflectance_L'+str(i_sensor)+'/',band_tags[band_tag],i_band,i,j)

            SR_band1[QC>0] = np.nan
            SR_band1[SR_band1==0] = np.nan
 
            month = month+[int(itm[5:7]) for itm in date]
            year = year+[int(itm[:4]) for itm in date]
            
            if i_sensor==sensor_list[0]:
                SR_band1_all_sensors = SR_band1.copy()
            else:
                SR_band1_all_sensors = np.concatenate([SR_band1_all_sensors,SR_band1],axis=2)
            
        
        SR_band1_annual = np.zeros([SR_band1.shape[0],SR_band1.shape[1],len(unique_year)])+np.nan
        month = np.array(month); year = np.array(year)
        for y in unique_year:
            filter_year =  (month>=6) & (month<=8) & (year==y)
            if sum(filter_year)>0:
                SR_band1_annual[:,:,y-min(unique_year)] = np.nansum(SR_band1_all_sensors[:,:,filter_year]>0,axis=2)
                
        print([np.nanquantile(SR_band1_annual,qt) for qt in [0.05,0.25,0.5,0.75,0.95]])   
#        SR_harmonized_allbands.append(SR_band1_annual) # including all bands





