#!/usr/bin/env python3

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import pickle

from numpy import ravel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


def get_latlon_lim(i,j): # limits of lat and lon for this tile
    square = np.array([[-151.2+j*0.2, 69.5-i*0.1],
                      [-151.2+j*0.2, 69.4-i*0.1],
                      [-151.0+j*0.2, 69.4-i*0.1],
                      [-151.0+j*0.2, 69.5-i*0.1]])
    return square

def get_latlon_2d(i,j,num_rows,num_cols):
    square = get_latlon_lim(i,j)
    lat = np.arange(np.max(square[:,1]),np.min(square[:,1]),-0.1/num_rows)-0.1/num_rows/2
    lon = np.arange(np.min(square[:,0]),np.max(square[:,0]),0.2/num_cols)+0.2/num_cols/2
    lat2d = np.transpose(np.tile(lat,[num_cols,1]))
    lon2d = np.tile(lon,[num_rows,1])
    return lat2d,lon2d 

def add_ndvi(training_data):
    training_data['ndvi'] = (training_data['nir'] - training_data['red'])/(training_data['nir'] + training_data['red'])
    return training_data

def LC_Prob(df_onetile,random_seeds,pft_names):

    Y = np.zeros([len(df_onetile),len(random_seeds)])# result of 500 models for all pixels in a tile and a year
    for value in random_seeds:
        print(value)
#        with open('RFmodels7_label/RFmodels' + str(value) + '.pkl', 'rb') as f:
#        with open('RFmodels7/RFmodels' + str(value) + '.pkl', 'rb') as f:
        with open('RFmodels578/RFmodels' + str(value) + '.pkl', 'rb') as f:
            rf_newparams = pickle.load(f)
        y_pred = rf_newparams.predict(df_onetile[band_list+['ndvi']]) # the ith trial of RF model (What is put into y_test>>)
        y_num = np.zeros([len(y_pred),])
        for ipft in range(len(pft_names)): # because numpy array works for numbers not characters
            y_num[y_pred==pft_names[ipft]] = ipft
        Y[:,value] = y_num.copy() # put y_pred as the ith column (for the ith trial of the RF model)

    Prob = np.transpose(np.array([np.mean(Y==ipft,axis=1) for ipft in range(len(pft_names))])) #[numpixel,numpft]
    return Prob

y = int(os.environ['SLURM_ARRAY_TASK_ID'])
#y = 0
#random_seeds = list(range(500))
#random_seeds = list(range(3))
random_seeds = list(range(10))
datapath = '/fs/scratch/PAS2094/ARF/'
#outpath = datapath+'Harmonized_Tiles/'
#outpath = datapath+'L7_Tiles/'
outpath = datapath+'L578_Tiles/'
unique_year = np.arange(2005,2021)

band_list = ['red', 'green','blue','swir2','swir1','nir','sr.b6']
#pft_names = ['GRA','SHD','SHE', 'MOS', 'SUB', 'LIV', 'FOR']
pft_names = ['SUB','GRA','FOR','SSE','LIV','SSD','MOS']
df_oneyear_alltiles = pd.DataFrame()

for i in range(7): #7
    for j in range(6): #6
        SR = np.load(outpath+'SR_harmonized_'+str(i)+str(j)+'.npy')
        lat2d,lon2d = get_latlon_2d(i,j,SR.shape[1],SR.shape[2])

        X_test = np.zeros([SR.shape[1]*SR.shape[2],len(band_list)]) 
        for k in range(len(band_list)):
            X_test[:,k] = SR[k,:,:,y].flatten()
        
        df_onetile = pd.DataFrame(np.column_stack([lat2d.flatten(),lon2d.flatten(),X_test]),columns=['latitude','longitude']+band_list)
        df_onetile = add_ndvi(df_onetile).dropna().reset_index()

        Prob = LC_Prob(df_onetile,random_seeds,pft_names)
        df_onetile = pd.concat([df_onetile,pd.DataFrame(Prob,columns=pft_names)],axis=1)

        df_oneyear_alltiles = pd.concat([df_oneyear_alltiles,df_onetile])

df_oneyear_alltiles[['latitude','longitude']+pft_names].reset_index().to_csv('LC578/LC_'+str(y+min(unique_year))+'.csv',index=False)
#df_oneyear_alltiles[['latitude','longitude']+pft_names].reset_index().to_csv('LC7_tr/LC_'+str(y+min(unique_year))+'.csv',index=False)
