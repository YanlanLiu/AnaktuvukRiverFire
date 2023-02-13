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
    training_data['ndvi'] = (0.0119 + 0.778*training_data['ndvi'] + 0.2017*(training_data['ndvi']**2))
    return training_data

y = int(os.environ['SLURM_ARRAY_TASK_ID'])
#y = 0

datapath = '/fs/scratch/PAS2094/ARF/'
#outpath = datapath+'Harmonized_Tiles/'
outpath = datapath+'L7_Tiles/'
unique_year = np.arange(2005,2021)

with open('RFmodels7.pkl', 'rb') as f:
    first_tree, tuned_tree, bag_tree, rf_tree, rf_newparams, gb_tree, gb_tree_tuned = pickle.load(f)

band_list = ['red', 'green','blue','swir2','swir1','nir','sr.b6']


df_oneyear_alltiles = pd.DataFrame()
df_lc = pd.DataFrame()

for i in range(7):
    for j in range(6):
        SR = np.load(outpath+'SR_harmonized_'+str(i)+str(j)+'.npy')
        lat2d,lon2d = get_latlon_2d(i,j,SR.shape[1],SR.shape[2])

        X_test = np.zeros([SR.shape[1]*SR.shape[2],len(band_list)])
        for k in range(len(band_list)):
            X_test[:,k] = SR[k,:,:,y].flatten()
        X_test = pd.DataFrame(X_test,columns=band_list)	

        coords = pd.DataFrame({'latitude':lat2d.flatten(),'longitude':lon2d.flatten()})
        df_oneyear_alltiles = pd.concat([df_oneyear_alltiles,X_test])
        df_lc = pd.concat([df_lc,coords])

df_oneyear_alltiles = add_ndvi(df_oneyear_alltiles)
df_oneyear_alltiles = pd.concat([df_oneyear_alltiles,df_lc],axis=1)
df_oneyear_alltiles = df_oneyear_alltiles.dropna()
print(df_oneyear_alltiles.head())
#df_lc['LC']= rf_newparams.predict(df_oneyear_alltiles.dropna())
df_oneyear_alltiles['LC']= gb_tree_tuned.predict(df_oneyear_alltiles[band_list+['ndvi']])
df_lc = df_oneyear_alltiles[['latitude','longitude','LC']]
df_lc.to_csv('LC7/LC_'+str(y+min(unique_year))+'.csv',index=False)  

