
# %load RF_training.py
#!/usr/bin/env python3
"""
Created on Mon Dec  5 14:53:29 2022

@author: yanlan
"""

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True,font_scale=1.75)
import pickle

from scipy.stats import randint
from numpy import ravel
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix  


def importance_plot(tree,label):
    feature_importance = tree.feature_importances_ # get the importance of each feature
    
    #calculate the relative feature importances 
    relative_importance = tree.feature_importances_ / max(tree.feature_importances_)
    
    # Save to a dataframe with two columns: One holding the names of the features, 
    # and one holding the associated relative importance of each feature.
    feat_df = pd.DataFrame({'feature':X_train.columns, 'importance':relative_importance})

    # Sort feat_df in order of importance
    feat_df = feat_df.sort_values(by='importance', ascending=True)
    
    plt.figure(figsize=(8, 7.5))

    # Create a bar chart. The widths of the bars should correspond to the importances, 
    # and y should correspond to the names of the features. 
    plt.barh(width=feat_df['importance'], y=feat_df['feature'])
    plt.xlabel('Relative feature importance')
    plt.title('Variable Importance Plot for' + str(tree));
    plt.savefig('Figures/Importance_'+label+'.png',dpi=300)

def add_ndvi(training_data):
    training_data['ndvi'] = (training_data['nir'] - training_data['red'])/(training_data['nir'] + training_data['red'])
#    training_data['ndvi'] = (0.0119 + 0.778*training_data['ndvi'] + 0.2017*(training_data['ndvi']**2))
    return training_data


training_data = pd.DataFrame()
for y in [2008,2009,2010,2011,2017]:
    df = pd.read_csv('Transects/training_data_'+str(y)+'.csv')
#     df = pd.read_csv('/Users/colettebrown/Google Drive/Shared drives/AnaktuvukRiverFire/OSC_Code/Transects7/training_data_'+str(y)+'.csv')

    training_data = pd.concat([training_data,df[df['red']>0]]) # to remove nan for reflectances

training_data = add_ndvi(training_data)

table = pd.pivot_table(training_data, values='latitude', index=['blue', 'green', 'red', 'nir', 'swir1', 'sr.b6', 'swir2', 'ndvi','Transect', 'PlotType'],
                    columns='local_growth_habit', aggfunc='count', fill_value=0)

table.reset_index(inplace=True)
table.loc[:,"FOR":"SUB"] = table.loc[:,"FOR":"SUB"].div(table.sum(axis=1), axis=0)
table['Class'] = table[['FOR','GRA', 'LIC', 'LIV', 'MOS', 'SHD', 'SSD', 'SSE', 'SUB']].idxmax(axis=1)


features = table[['blue', 'green', 'red', 'nir', 'swir1', 'sr.b6', 'swir2', 'ndvi']]
target = table[['Class']]
target = ravel(target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=2022)

table['weights'] = table['Class'].map({'SUB': 0.3640776699,
                                       'GRA': 0.3947368421,
                                       'FOR': 9.375,
                                       'SSE': 2.027027027,
                                       'LIV': 6.8181818182,
                                       'SSD': 4.6875,
                                       'MOS': 1.4423076923})
sample = table.sample(n=520, weights='weights', replace=True, random_state=1)
features = sample[['blue', 'green', 'red', 'nir', 'swir1', 'sr.b6', 'swir2', 'ndvi']]
target = sample[['Class']]
target = ravel(target)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=111)
plt.figure()
plt.hist(y_train, label='Train', alpha=0.2)
# plt.hist(y_val, label = 'Validation', alpha=0.4)
plt.hist(y_test, label = 'Test', alpha=0.2)
plt.legend()
plt.title("Distribution of Class for Training, Validation, and Test Data")
plt.savefig('./Figures/Tr_val_dist.png',dpi=300,bbox_inches='tight')

print(table.groupby(['Class']).count())


rf_tree = RandomForestClassifier(random_state=2021, n_estimators=150)
rf_tree.fit(X_train, y_train)

rf_train_score = rf_tree.score(X_train, y_train)
rf_test_score = rf_tree.score(X_test, y_test)

print('Train Score: ', rf_train_score)
print('Test Score: ', rf_test_score)

param_dist = {'n_estimators':range(1, 500)} # specify a dictionary with two parameters and the range of values over which you'd like to choose values

rf_tree_search = RandomizedSearchCV(rf_tree,param_distributions=param_dist,cv=7, n_iter=10, random_state = 2021)

rf_tree_search.fit(X_train, y_train)

print(rf_tree_search.best_params_)

rf_newparams = RandomForestClassifier(random_state=2021, **rf_tree_search.best_params_)
rf_newparams.fit(X_train, y_train)

rf_newparams_train_score = rf_newparams.score(X_train, y_train)
rf_newparams_test_score = rf_newparams.score(X_test, y_test)

print('Train Score: ', rf_newparams_train_score)
print('Validation Score: ', rf_newparams_test_score)


scores = []
models = [rf_tree, rf_newparams]
for i in models:
    scores.append(i.score(X_test, y_test))
    print('Test Score: ', i.score(X_test, y_test))


y_pred = rf_newparams.predict(X_test)
print(confusion_matrix(y_test,y_pred))  
print('Classification Report for Tuned Random Forest')
print(classification_report(y_test,y_pred)) 


# random_seeds = list(range(2))
random_seeds = list(range(500))
rf_feature_imp_list = []
rf_val_scores = []
rf_train_scores = []


for value in random_seeds:
#     X, X_test, y, y_test = train_test_split(features, target, test_size=0.2, random_state=random_seeds[value])
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.25, random_state = random_seeds[value])
    rf_newparams_forloop = RandomForestClassifier(random_state=2021, n_estimators=300)
    rf_newparams_forloop.fit(X_train, y_train)
    rf_newparams_train_score = rf_newparams_forloop.score(X_train, y_train)
    rf_newparams_val_score = rf_newparams_forloop.score(X_val, y_val)
    rf_feature_imp_list.append(rf_newparams_forloop.feature_importances_)
    rf_val_scores.append(rf_newparams_val_score)
    rf_train_scores.append(rf_newparams_train_score)
    
    with open('RFmodels/RFmodels' + str(value) + '.pkl', 'wb') as f:
        pickle.dump(rf_newparams_forloop, f)
        


rf_feature_importance = pd.DataFrame(rf_feature_imp_list, columns=['blue', 'green', 'red', 'nir', 'swir1', 'sr.b6', 'swir2', 'ndvi'])
plt.figure()
rf_feature_importance.boxplot()
plt.xlabel('Feature')
plt.ylabel('Importance Values')
plt.title('Feature Importance Value Distributions (RandomForest, 500)')
plt.savefig('Figures/Feature_importance.png',dpi=300,bbox_inches='tight')

fig, ax = plt.subplots()
sns.kdeplot(rf_feature_importance['blue'], ax=ax, label = 'blue')
sns.kdeplot(rf_feature_importance['green'], ax=ax, label = 'green')
sns.kdeplot(rf_feature_importance['red'], ax=ax, label = 'red')
sns.kdeplot(rf_feature_importance['nir'], ax=ax, label = 'nir=')
sns.kdeplot(rf_feature_importance['swir1'], ax=ax, label = 'swir1')
sns.kdeplot(rf_feature_importance['sr.b6'], ax=ax, label = 'sr.b6')
sns.kdeplot(rf_feature_importance['swir2'], ax=ax, label = 'swir2')
sns.kdeplot(rf_feature_importance['ndvi'], ax=ax, label = 'ndvi').set(title='KDE Plots of Variable Importance (RandomForest, 1000)')
plt.legend(bbox_to_anchor=(1.05,1.05))
plt.savefig('Figures/KDE_var_importance.png',dpi=300,bbox_inches='tight')



with open('RFmodels.pkl', 'wb') as f: 
    pickle.dump((rf_tree, rf_newparams), f)


rf_scores = pd.DataFrame(rf_val_scores, columns=['Validation Scores'])
rf_scores['Training Scores'] = rf_train_scores
fig, ax = plt.subplots()
sns.kdeplot(rf_scores['Validation Scores'], ax=ax).set(title='KDE Plots of Validation (RandomForest, 500)')
plt.axvline(np.mean(rf_scores['Validation Scores']), 0,15, 
            label='Mean:' + str(np.mean(rf_scores['Validation Scores'])), 
            color='Black', ls='--')
plt.legend(loc='best')
plt.savefig('Figures/Validation_scores.png',dpi=300,bbox_inches='tight')
    

