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


arrayid = 40
#arrayid = 20
i = arrayid//6
j = arrayid-i*6

datapath = '/fs/scratch/PAS2094/ARF/'
i_sensor = 7
#fname = '30L7QA_2ee-export-video11658537125461241173.mp4'
#nid = imageio.get_reader(datapath+'SurfaceReflectance_L7/QC_PIXEL/'+fname,'ffmpeg')
#img = nid.get_data(1)[:]
#print(img)
#print(np.unique(img))
#plt.imshow(img[:,:,0]);plt.colorbar();plt.savefig('tmp.png',dpi=300,bbox_inches='tight')

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
    print(date)
    #cloud_cover_series = pd.Series(img_csv['cloud'])
    sr_3d = np.zeros([img0.shape[0],img0.shape[1],len(img_csv)])
    
    for i_time in range(len(img_csv)): 
        sr_3d[:,:,i_time] = np.array(nid.get_data(i_time)[:,:,i_band],dtype=float)/255
        # returns a stack of images that is one array for every time stamp 
    nid.close()

    if i_band==0 and Btag == band_tags[0]:
        QC = get_qc_pixel(srpath+'QC_PIXEL/',i,j,img0.shape[0],img0.shape[1],len(img_csv))
    
    else:
        QC = 0 
    return sr_3d, date, QC


band_tags = ['B321', 'B754', 'B6']
unique_year = np.arange(2005,2022)
band_tag = 0
i_band=0
SR_band1, date, QC = get_3d_sr(datapath+'SurfaceReflectance_L'+str(i_sensor)+'/',band_tags[band_tag],i_band,i,j)
print(QC.shape)
plt.figure()
plt.imshow(QC[:,:,0]); plt.colorbar()
plt.savefig('QC_t1.png',dpi=300,bbox_inches='tight')
#plt.figure()
#plt.imshow(SR_band1[:,:,0]);plt.colorbar()
#plt.savefig('band1.png',dpi=300,bbox_inches='tight')

