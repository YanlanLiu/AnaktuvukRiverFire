#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:53:29 2022

@author: yanlan
"""

import numpy as np
import pandas as pd
import glob
import os
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])

def get_latlon_lim(i,j): # limits of lat and lon for this tile
    square = np.array([[-151.2+j*0.2, 69.5-i*0.1],
                      [-151.2+j*0.2, 69.4-i*0.1],
                      [-151.0+j*0.2, 69.4-i*0.1],
                      [-151.0+j*0.2, 69.5-i*0.1]])
    return square

def find_tile(lat,lon):
    i,j = int((69.5-lat)//0.1), int((lon+151.2)//0.2)
    return i,j

def find_rc(square,SR,lat0,lon0):
    num_rows,num_cols = SR.shape[1],SR.shape[2]
    lat = np.arange(np.max(square[:,1]),np.min(square[:,1]),-0.1/num_rows)-0.1/num_rows/2
    lon = np.arange(np.min(square[:,0]),np.max(square[:,0]),0.2/num_cols)+0.2/num_cols/2
    r = np.argmin(np.abs(lat-lat0))
    c = np.argmin(np.abs(lon-lon0))
    return r,c


datapath = '/fs/scratch/PAS2094/ARF/'
#outpath = datapath+'Harmonized_Tiles/'
outpath = datapath+'L7_Tiles/'
transects_path = datapath+'FieldData/'
unique_year = np.arange(2005,2021)
tr_list = glob.glob(transects_path+"point*.csv")
band_list = ['red', 'green','blue','swir2','swir1','nir','sr.b6']

#training_data_allyears = pd.DataFrame()
tr_list.sort()

for tr in tr_list[arrayid:arrayid+1]: # loop over all the transect csv files
    print(tr)
    sid = tr[::-1].find('_') # location of the last '_'
    year = int(tr[-sid:-sid+4])
    df = pd.read_csv(tr)  
    yid = year-unique_year[0]
    sr_list = np.zeros([len(df),len(band_list)])+np.nan
    for k in range(len(df)):
        lat0,lon0 = df['latitude'].iloc[k], df['longitude'].iloc[k]
        i,j = find_tile(lat0,lon0)
        if i>=0 and j>=0:
            square = get_latlon_lim(i,j)
            SR = np.load(outpath+'SR_harmonized_'+str(i)+str(j)+'.npy')
            r,c = find_rc(square,SR,lat0,lon0)
            sr_list[k,:] = SR[:,r,c,yid].flatten()
        else:
            print(i,j,lat0,lon0)
    training_data = pd.concat([df,pd.DataFrame(sr_list,columns=band_list)],axis=1)
    #training_data_allyears = pd.concat([training_data_allyears,training_data.dropna()])
#    training_data.reset_index().drop(columns=['index']).to_csv('Transects/training_data_'+str(year)+'.csv',index=False)
    training_data.reset_index().drop(columns=['index']).to_csv('Transects7/training_data_'+str(year)+'.csv',index=False)

