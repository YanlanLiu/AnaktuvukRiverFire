#!/usr/bin/env python
# coding: utf-8



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
    plt.savefig('Figures578/Importance_'+label+'.png',dpi=300)

def add_ndvi(training_data):
    training_data['ndvi'] = (training_data['nir'] - training_data['red'])/(training_data['nir'] + training_data['red'])
    return training_data

# import the training data and combine all years
training_data = pd.DataFrame()
for y in [2008,2009,2010,2011,2017]:
    df = pd.read_csv('Transects578/training_data_'+str(y)+'.csv')
#    df = pd.read_csv('/Users/colettebrown/Google Drive/Shared drives/AnaktuvukRiverFire/OSC_Code/BKP/Transects7/training_data_'+str(y)+'.csv')
    training_data = pd.concat([training_data,df[df['red']>0]]) # to remove nan for reflectances
    training_data = add_ndvi(training_data)
    
def combining_points(training_data, class_hierarchy):
    if class_hierarchy == "GRA":
        hierarchy = {"SHD": 1, "GRA": 2, "SSD": 3, "SSE": 4, "FOR": 5, "LIV": 6, "LIC": 7, "MOS": 8, "SUB" : 9}
    elif class_hierarchy == "SSD":
        hierarchy = {"SHD": 1, "SSD": 2, "GRA": 3, "SSE": 4, "FOR": 5, "LIV": 6, "LIC": 7, "MOS": 8, "SUB" : 9}
        
    training_data['Hierarchy'] = training_data["local_growth_habit"].map(hierarchy)

    # sort rows by year, transect, point, and hierarchy and drops duplicates keeping the first row for each unique combo
    training_data = training_data.sort_values(by=["year", "Transect", "Point", "Hierarchy"]).drop_duplicates(subset=["year", "Transect", "Point"], keep="first")

    # create a new column with the class label based on the hierarchical rank
    # training_data["Class"] = training_data["local_growth_habit"].map(hierarchy)
    training_data["Class"] = training_data["Hierarchy"].apply(lambda x: [k for k, v in hierarchy.items() if v == x][0])
    training_data = training_data.drop('Hierarchy', axis=1)
    
    return(training_data)

def group_transect_pixel(training_data, grouped_by, threshold):
    if grouped_by == "Transect":
        table_grouped_transects = pd.pivot_table(training_data, values='latitude', index=['year','Transect', 'PlotType'],
                    columns='local_growth_habit', aggfunc='count', fill_value=0)
        table_grouped_transects.reset_index(inplace=True)
        table_grouped_transects.loc[:,"FOR":"SUB"] = table_grouped_transects.loc[:,"FOR":"SUB"].fillna(0).div(table_grouped_transects.loc[:, "FOR":"SUB"].sum(axis=1, numeric_only=True), axis=0)*100
        
        #this returns the table of just "year, transect, plottype, list of growth forms with percent cover in each transect"
        # how do we create the threshold and re-assign "training_data" in this function using this dataframe
        # we create unique table names for the table that is being compared depending on pixel vs. transect 
        for row in range(len(table_grouped_transects)):
            year = table_grouped_transects.iloc[row]['year']
            transect = table_grouped_transects.iloc[row]['Transect']
            if table_grouped_transects.iloc[row]['SHD'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'SHD'
            elif table_grouped_transects.iloc[row]['GRA'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'GRA'
            elif table_grouped_transects.iloc[row]['SSD'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'SSD'
            elif table_grouped_transects.iloc[row]['SSE'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'SSE'
            elif table_grouped_transects.iloc[row]['FOR'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'FOR'
            elif table_grouped_transects.iloc[row]['LIV'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'LIV'
            elif table_grouped_transects.iloc[row]['MOS'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'MOS'
            elif table_grouped_transects.iloc[row]['SUB'] >= threshold:
                training_data.loc[(training_data['year'] == year) & (training_data['Transect'] == transect), 'Class'] = 'SUB'
        
        
    pixel_training_data = pd.pivot_table(training_data, values='latitude', index=['blue', 'green', 'red', 'nir', 'swir1', 'sr.b6', 'swir2', 'ndvi','Transect', 'PlotType'],
                    columns='Class', aggfunc='count', fill_value=0)
    
    pixel_training_data.reset_index(inplace=True)
    
    # find the columns to locate between "PlotType" and "Class"
    plot_type_loc = pixel_training_data.columns.get_loc('PlotType')
    cols_between = pixel_training_data.columns[plot_type_loc+1:].to_list()
    
    # find percent cover of each class in the pixel
    pixel_training_data.loc[:,cols_between] = pixel_training_data.loc[:,cols_between].fillna(0).div(pixel_training_data.loc[:,cols_between].sum(axis=1, numeric_only=True), axis=0)*100
    

    pixel_training_data['Class'] = pixel_training_data[cols_between].idxmax(axis=1)
    

    if grouped_by == 'Pixel':
        for row in range(len(pixel_training_data)):
            if pixel_training_data.iloc[row]['SHD'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'SHD'
            elif pixel_training_data.iloc[row]['GRA'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'GRA'
            elif pixel_training_data.iloc[row]['SSD'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'SSD'
            elif pixel_training_data.iloc[row]['SSE'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'SSE'
            elif pixel_training_data.iloc[row]['FOR'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'FOR'
            elif pixel_training_data.iloc[row]['LIV'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'LIV'
            elif pixel_training_data.iloc[row]['MOS'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'MOS'
            elif pixel_training_data.iloc[row]['SUB'] >= threshold:
                pixel_training_data.loc[row, 'Class'] = 'SUB'        

    return(pixel_training_data)

training_data = combining_points(training_data, "GRA") # either "GRA" or "SSD"
training_data = group_transect_pixel(training_data, grouped_by='Pixel', threshold=20) # either "Pixel" or "Transect"
#training_data = group_transect_pixel(training_data, grouped_by='Transect', threshold=20)
class_counts = training_data['Class'].value_counts()
class_weights = 1/(class_counts / len(training_data))

training_data['weights'] = (training_data['Class'].map(class_weights))/((training_data['Class'].map(class_weights)).sum())

sample = training_data.sample(n=520, weights='weights', replace=True, random_state=1)

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
plt.savefig('./Figures578/Tr_val_dist.png',dpi=300,bbox_inches='tight')


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


#random_seeds = list(range(2))
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
    
    with open('./RFmodels578/RFmodels' + str(value) + '.pkl', 'wb') as f:
#    with open('./RFmodels7_label/RFmodels' + str(value) + '.pkl', 'wb') as f:
        pickle.dump(rf_newparams_forloop, f)
        


rf_feature_importance = pd.DataFrame(rf_feature_imp_list, columns=['blue', 'green', 'red', 'nir', 'swir1', 'sr.b6', 'swir2', 'ndvi'])
plt.figure()
rf_feature_importance.boxplot()
plt.xlabel('Feature')
plt.ylabel('Importance Values')
plt.title('Feature Importance Value Distributions (RandomForest, 500)')
plt.savefig('Figures578/Feature_importance.png',dpi=300,bbox_inches='tight')

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
plt.savefig('Figures578/KDE_var_importance.png',dpi=300,bbox_inches='tight')



with open('RFmodels578.pkl', 'wb') as f: 
    pickle.dump((rf_tree, rf_newparams), f)


rf_scores = pd.DataFrame(rf_val_scores, columns=['Validation Scores'])
rf_scores['Training Scores'] = rf_train_scores
fig, ax = plt.subplots()
sns.kdeplot(rf_scores['Validation Scores'], ax=ax).set(title='KDE Plots of Validation (RandomForest, 500)')
plt.axvline(np.mean(rf_scores['Validation Scores']), 0,15, 
            label='Mean:' + str(np.mean(rf_scores['Validation Scores'])), 
            color='Black', ls='--')
plt.legend(loc='best')
plt.savefig('Figures578/Validation_scores.png',dpi=300,bbox_inches='tight')
    







