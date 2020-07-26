# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV 


def fetch_data_and_clean():
        
    df_energy = pd.read_csv("Usecase1_Dataset.csv")
    
    df_energy = df_energy[df_energy["ENERGY STAR Score"] != "Not Available"].reset_index().drop(["index","Order"],axis=1)
    df_energy = df_energy.replace("Not Available",np.nan)
    
    
    missing_data_perc = dict()
    for i in df_energy.columns:
        missing_data_perc[i] = len(df_energy[~df_energy[i].isnull()])/len(df_energy)
    
    missing_data_perc = pd.Series(missing_data_perc) * 100
    missing_data_perc_filter_40_perc_missing = missing_data_perc[missing_data_perc.sort_values(ascending=False) > 60]
    
    
    ##########################
    duplicate_location_columns = ['Property Name',
       'Parent Property Name','Street Number','Street Name',
       'NYC Borough, Block and Lot (BBL) self-reported',
       'NYC Building Identification Number (BIN)', 'Address 1 (self-reported)',
       'Street Name', 'Borough','Latitude', 'Longitude',
       'Community Board', 'Council District', 'Census Tract', 'NTA']
    
    #Carrying only relavant columns having more than 40 perc not null values
    df_energy = df_energy[list(missing_data_perc_filter_40_perc_missing.index)]
    
    #removing duplicate location columns as stated before
    df_energy = df_energy.drop(duplicate_location_columns,axis=1) 
    
    # As per Data set, segregating  BBL code to 3 columns for NYC Borough, block and lot codes which along with
    #pin code will give info about building and locality

    df_energy["NYC Borough"] = df_energy["BBL - 10 digits"].apply(lambda x:str(x)[0:1])
    df_energy["NYC Block"] = df_energy["BBL - 10 digits"].apply(lambda x:str(x)[1:6])
    df_energy["NYC Lot"] = df_energy["BBL - 10 digits"].apply(lambda x:str(x)[6:10])
    
    #dropping BBL code as it is no longer necessary
    df_energy = df_energy.drop(['BBL - 10 digits','Release Date'],axis=1)
    
    #Reorganising Dataset
    df_energy = pd.concat([df_energy.drop('ENERGY STAR Score',axis=1),df_energy["ENERGY STAR Score"]],axis=1)
    
    #Replacing standalone bulding parent property id with property id.
    for i in range(len(df_energy)):
        if(df_energy["Parent Property Id"][i] == "Not Applicable: Standalone Property"):
            df_energy["Parent Property Id"][i] = df_energy["Property Id"][i]  
    
    #Organising Category columns under one list for ease
    cat_columns = ['Primary Property Type - Self Selected',
                   'List of All Property Use Types at Property',
                   'Largest Property Use Type','Metered Areas (Energy)',
                   'Water Required?','Metered Areas  (Water)',
                   'DOF Benchmarking Submission Status']
    
    #Transforming categories to numbers for regression
    for col in cat_columns:
        try:
            df_energy[col] = LabelEncoder().fit_transform(df_energy[col])
        except:
            df_energy[col] = df_energy[col].fillna(value=" ")
            df_energy[col] = LabelEncoder().fit_transform(df_energy[col])  
            
    df_energy_fliter = df_energy.copy()
    
    ####################Data Cleaning Begins Here ####################################
    #below code was used to find some junk values rows which were troubling.
    #df_energy_fliter[df_energy_fliter.eq('\u200b').any(1)].T
    df_energy_fliter = df_energy_fliter.drop([95,190,9630,9631],axis=0).reset_index().drop('index',axis=1)
    
    #Cleaning Postal code as some junk or multiple values were there after 6 character in some rows
    df_energy_fliter["Postal Code"] = df_energy_fliter["Postal Code"].apply(lambda x:str(x)[:5]) 
    
    #replacing null values with zeroes for row for integer type columns for now
    df_energy_fliter = df_energy_fliter.replace(np.nan,0)
    
    df_energy_fliter = df_energy_fliter.astype(float)
    
    
    ############Replacing some nulls with mean#############################
    for i in range(len(df_energy_fliter)):
        if((df_energy_fliter['Water Intensity (All Water Sources) (gal/ft²)'][i] == 0) 
           or (df_energy_fliter['Water Intensity (All Water Sources) (gal/ft²)'][i] == 0)):
            
            cat_no = df_energy_fliter["Largest Property Use Type"][i]
            
            df_energy_fliter['Water Intensity (All Water Sources) (gal/ft²)'][i] = df_energy_fliter[df_energy_fliter["Largest Property Use Type"] == cat_no]['Water Intensity (All Water Sources) (gal/ft²)'].mean() 
            
            df_energy_fliter['Water Use (All Water Sources) (kgal)'][i] = df_energy_fliter[df_energy_fliter["Largest Property Use Type"] == cat_no]['Water Use (All Water Sources) (kgal)'].mean() 
        
    #For Future Reference   
    '''
    corr_matrix = df_energy_fliter.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Find index of feature columns with correlation greater than 0.90
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    # Drop features 
    df_energy_fliter_non_corr =df_energy_fliter.drop(df_energy_fliter[to_drop], axis=1)
    '''
    return(df_energy_fliter)


def train_test_model():
    
    df_energy_fliter = fetch_data_and_clean()
    
    ##########Train Test Split##################
    X_train, X_test, y_train, y_test = train_test_split(df_energy_fliter.drop(["ENERGY STAR Score"]
                                                                           ,axis=1),df_energy_fliter["ENERGY STAR Score"] 
                                                                           ,test_size = 0.2)
    
    model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                               colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=-1,
                               importance_type='gain', interaction_constraints='',
                               learning_rate=0.1, max_delta_step=0, max_depth=5,
                               min_child_weight=5, monotone_constraints='()',
                               n_estimators=200, n_jobs=4, nthread=4, num_parallel_tree=1,
                               random_state=27, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                               seed=27, subsample=0.8, tree_method='exact', validate_parameters=1,
                               verbosity=None)
    
    
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, predictions_train)
    mse_train = mean_squared_error(y_train, predictions_train)
    
    mae_test = mean_absolute_error(y_test, predictions_test)
    mse_test = mean_squared_error(y_test, predictions_test)
    
    print('XGBoost Performance on the train set: MAE = %0.4f' % mae_train)
    print('XGBoost Performance on the train set: MSE = %0.4f' % mse_train)
    
    print('XGBoost Performance on the test set: MAE = %0.4f' % mae_test)
    print('XGBoost Berformance on the test set: MSE = %0.4f' % mse_test)
        
    '''
    param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'n_estimators':range(100,500,100)}
    
    gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1,
                            gamma=0, subsample=0.8, colsample_bytree=0.8,
                                 nthread=4, scale_pos_weight=1, seed=27), 
                            param_grid = param_test1,n_jobs=4,iid=False, cv=5)
    
    gsearch1.fit(df_energy_fliter.drop(["ENERGY STAR Score"],axis=1),df_energy_fliter["ENERGY STAR Score"])
    gsearch1.best_params_, gsearch1.best_score_
    '''
    
    df_output = pd.concat([pd.Series(y_test).reset_index().drop('index',axis=1),pd.Series(predictions_test).reset_index().drop('index',axis=1)],axis=1)
    df_output = df_output.rename({0:"Predicted_Score"},axis=1)
    
    return(df_output)


df_output = train_test_model()    
    