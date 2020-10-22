#!/usr/bin/env python
# coding: utf-8


import datetime
import calendar
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   RobustScaler, MaxAbsScaler,
                                   PowerTransformer, QuantileTransformer,
                                   OneHotEncoder, OrdinalEncoder,
                                   KBinsDiscretizer)

class Handle_Datatype(BaseEstimator, TransformerMixin):

    def __init__(self, target, ml_usecase, categorical_features=[], numerical_features=[], time_features=[],
                 features_to_drop=[]):

        self.replacement = {}
        self.id_columns = []
        self.target = target
        self.ml_usecase = ml_usecase
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.time_features = time_features
        self.features_to_drop = features_to_drop
        
    def fit(self, dataset, y=None):
        data = dataset.copy()
        
        # drop user given features
        data.drop(columns=self.features_to_drop, errors='ignore', inplace=True)
        
        # replace inf in NaN values
        data.replace([np.inf, -np.inf], np.NaN, inplace=True)
        
        # try to clean columns names
        data.columns = data.columns.str.replace(r'[\,\}\{\]\[\:\"\']','')
        
        # try to convert category features into integer if possible
        for i in data.select_dtypes(include=['object']).columns:
            try:
                data[i] = datap[i].astype('int64')
            except:
                None
        # convert boolean and category datatypes into categirical features
        for i in data.select_dtypes(include=['bool', 'category']).columns:
            data[i] = data[i].astype('object')
            
        # handling datetime features
        for i in data.select_dtypes(include=['object']).drop(self.target, axis=1, errors='ignore').columns:
            try:
                data[i] = pd.to_datetime(data[i], infer_datetime__format=True, utc=False, errors='raise')
            except:
                continue
        # convert given numerical into numerical features
        for i in self.categorical_features:
            try:
                data[i] = data[i].apply(str)
            except:
                None
                
        # convert given categorical into categorical features
        for i in self.numerical_features:
            try:
                data[i] = data[i].apply('float64')
            except:
                None
                
        # convert given datetime features into datetime features
        for i in self.time_features:
            try:
                data[i] = pd.to_datetime(data[i], infer_datetime__format=True, utc=False, errors='raise').astype('datetime[ns]')
            except:
                None
        
        # time feature datatype hardcoding
        for i in data.select_dtypes(include=['datetime64']).columns:
            data[i] = data[i].astype('datetime[ns]')
        
        # remove duplicates
        data = data.loc[:, ~data.columns.duplicated()]
        # remove NaN's
        data.dropna(axis=0, how='all', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)

        # remove the row if target column have any nan
        data = data[~data[self.target].isnull()]
        
        return data
    
    def transform(self, dataset, y=None):
        data = dataset.copy()
        
        # drop user given features
        data.drop(self.features_to_drop, errors='ignore', inplace=True)
        
        # make sure all the columns are of string
        data.columns = [str(i) for i in data.columns]
        
        # reaplace inf into NaNs
        data.replace([np.inf, -np.inf], np.NaN, inplace=True)
        
        for i in self.final_training_columns:
            if i not in data.columns:
                print("Test data does not have columns" + str(i)+" which was used for training")
        
        data = data[self.final_training_columns]
        
        return data
    
    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        data = self.fit(data)
        
        # we need to treat the target variale if any accordingly
        if (self.ml_usecase=='classification') & (data[self.target].dtypes=='object'):
            le = LabelEncoder()
            data[self.target] = le.fit_transform(np.array(data[self.target]))
            
            # now get the replacement dictionary
            rev = le.inverse_transform(range(0, len(le.classes_)))
            rep = np.array(range(0, len(le.classes_)))
            for v, k in zip(rev, rep):
                self.replacement[v] = k
            
            # for testing purpose
            self.final_training_columns = data.drop(self.target, axis=1).columns
            
            return data

class Handle_Missing(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable, numeric_strategy, categorical_strategy):
        self.target = target_variable
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        
    def fit(self, dataset, y=None):
        def zeros(x):
            return 0
        
        data = dataset.copy()
        
        # Numerical features
        if self.numeric_strategy == 'mean':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['int64', 'float64']).apply(np.nanmean)
        elif self.numeric_strategy == 'median':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['int64', 'float64']).apply(np.nanmedian)
        elif self.numeric_strategy == 'std':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['int64', 'float64']).apply(np.nanstd)
        elif self.numeric_strategy == 'mode':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['int64', 'float64']).apply(np.nanmedian)
        elif self.numeric_strategy == 'zero':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['int64', 'float64']).apply(zeros)
        else:
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['int64', 'float64']).apply(zeros)
        
        self.numeric_columns = data.drop(self.target, axis=1).select_dtypes(include=['float64', 'int64']).columns
        
        # Categorical features
        if self.categorical_strategy == 'most frequent':
            self.categorical_columns = data.drop(self.target, axis=1).select_dtypes(include=['object']).columns
            self.categoric_stats = pd.DataFrame(columns=self.categorical_columns)
            for i in self.categoric_stats.columns:
                self.categoric_stats.loc[0, i] = data[i].value_counts().index[0]
        else:
            self.categorical_columns = data.drop(self.target, axis=1).select_dtypes(include=['object']).columns
        
        # datatime features
        self.time_columns = data.drop(self.target, axis=1).select_dtypes(include=['datetime64[ns]']).columns
        self.time_stats = pd.DataFrame(columns=self.time_columns)
        for i in self.time_columns:
            self.time_stats.loc[0, i] = data[i].value_counts().index[0]
            
        return data
    
    def transform(self, dataset, y=None):
        data = dataset.copy()
        
        # for numeric
        for i, s in zip(data[self.numeric_columns].columns, self.numeric_stats):
            data[i].fillna(s, inplace=True)
        
        # for categoric
        if self.categorical_strategy == 'most frequent':
            for i in self.categoric_stats.columns:
                
                data[i] = data[i].fillna(self.categoric_stats.loc[0, i])
                data[i] = data[i].apply(str)
        else:
            for i in self.categorical_columns:
                data[i].fillna("not available", inplace=True)
                data[i] = data[i].apply(str)
                
        # for datetime
        for i in self.time_stats.columns:
            data[i].fillna(self.time_stats.loc[0, i], inplace=True)
        
        return data
    
    def fit_transform(self, dataset, y=None):
        data= dataset.copy()
        data = self.fit(data)
        return self.transform(data)
    

class Handle_Zero_NearZero_Variance(BaseEstimator, TransformerMixin):
    '''
        It eleminates the features having zero or near zero variance
        Near Zero Variance is ddetermined by 
        - 1. count of unique points divided by the total length of the feature has to be lower than a prespecified threshold
        - 2. Count of Most common point divided by the count of second most common point in the features is greater than a pre specified threshold
        Once both are met than feature is dropped
    '''
    
    def __init__(self, target, threshold_1=0.1, threshold_2=20):
        self.target = target
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.to_drop = []
        
    def fit(self, dataset, y=None):
        data = dataset.copy()
        self.sample_len = len(data[self.target])
        for i in data.drop(self.target, axis=1).columns:
            u = pd.DataFrame(data[i].value_counts()).sort_values(by=i, ascending=False, inplace=False)
            first = len(u)/self.sample_len
            # Below : it means f column is non variance automaticaly make the number big to drop it 
            if len(u[i]) == 1:
                second = 100
            else:
                second = u.iloc[0, 0] / u.iloc[1, 0]
                
            # If both conditions are met -> Drop the coumn
            if (first <= self.threshold_1) and (second >= self.threshold_2) and (i[-10:] != '_surrogate'):
                self.to_drop.append(i)
            if (second == 100) and (i[-10:] != '_surrogate'):
                self.to_drop.append(i)
                
    def transform(self, dataset, y=None):
        data = dataset.copy()
        data.drop(self.to_drop, axis=1, inplace=True)
        return data
    
    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        self.fit(data)
        return self.transform(data)

class Group_Similiar_Features(BaseEstimator, TransformerMixin):
    def __init__(self, group_name=[], list_of_grouped_features=[[]]):
        self.group_name = group_name
        self.list_of_similar_features = list_of_grouped_features
            
    def fit(self, dataset, y=None):
        return None
    
    def transform(self, dataset, y=None):
        data =  dataset.copy()
        if len(self.list_of_similar_features) > 0 and np.array(self.list_of_similar_features).shape[0] >= 1:
            for f, g in zip(self.list_of_similar_features, self.group_name):
                data[g + '_Min'] = data[f].apply(np.min, 1)
                data[g + '_Max'] = data[f].apply(np.max, 1)
                data[g + '_Mean'] = data[f].apply(np.mean, 1)
                data[g + '_Median'] = data[f].apply(np.median, 1)
                data[g + '_Mode'] = stats.mode(data[f], 1)[0]
                data[g + '_Std'] = data[f].apply(np.std, 1)
            return data
        else:
            return data
        
    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        self.fit(data)
        return self.transform(data)

class Scaling_and_Power_Transformation(BaseEstimator, TransformerMixin):
    def __init__(self, target, function_to_apply='zscore', random_state_quantile=42):
        self.target = target
        self.function_to_apply = function_to_apply
        self.random_state_quantile = random_state_quantile
        
    def fit(self, dataset, y=None):
        data = dataset.copy()
        self.numeric_features = data.drop(self.target, axis=1, errors='ignore').select_dtypes(include=['int64','float64']).columns
        
        # if there is any numerical feature
        if len(self.numeric_features) > 0:
            if self.function_to_apply == 'zscore':
                self.scale_and_power = StandardScaler()
                self.scale_and_power.fit(data[self.numeric_features])
            elif self.function_to_apply == 'minmax':
                self.scale_and_power = MinMaxScaler()
                self.scale_and_power.fit(data[self.numeric_features])
            elif self.function_to_apply == 'yj':
                self.scale_and_power = PowerTransformer(method='yeo-johnson', standardize=True)
                self.scale_and_power.fit(data[self.numeric_features])
            elif self.function_to_apply == 'quantile':
                self.scale_and_power = QuantileTransformer(random_state=self.random_state_quantile,
                                                          output_distribution='normal')
                self.scale_and_power.fit(data[self.numeric_features])
            elif self.function_to_apply == 'robust':
                self.scale_and_power = RobustScaler()
                self.scale_and_power.fit(data[self.numeric_features])
            elif self.function_to_apply == 'maxabs':
                self.scale_and_power = MaxAbsScaler()
                self.scale_and_power.fit(data[self.numeric_features])
            else:
                return None
        else:
            None
            
    
    def transform(self, dataset, y=None):
        data = dataset.copy()
        if len(self.numeric_features) > 0:
            self.data_t = pd.DataFrame(self.scale_and_power.transform(data[self.numeric_features]))
            self.data_t.index = data.index
            self.data_t.columns = self.numeric_features
            for i in self.numeric_features:
                data[i] = self.data_t[i]
                
            return data
        else:
            return data
    
    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        self.fit(data)
        return self.transform(data)

class Target_Transformation(BaseEstimator, TransformerMixin):
    def __init__(self, target, function_to_apply='bc'):
        self.target = target
        self.function_to_apply = function_to_apply
        if self.function_to_apply == 'bc':
            self.function_to_apply = 'box-cox'
        else:
            self.function_to_apply = 'yeo-johnson'
    
    def fit(self, dataset, y=None):
        return None
    
    def transform(self, dataset, y=None):
        return dataset
    
    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        
        # if target column has zero or negative values then auto use yj method
        if any(data[self.target] <= 0):
            self.function_to_apply = 'yeo-johnson'
        self.p_transform_target = PowerTransformer(method=self.function_to_apply)
        
        data[self.target] = self.p_transform_target.fit_transform(np.array(data[self.target]).reshape(-1, 1))
        
        return data

class Make_Time_Features(BaseEstimator, TransformerMixin):
    def __init__(self, time_features=[], list_of_features=['month', 'weekday', 'is_month_end', 'is_month_start', 'hour']):
        self.time_features = time_features
        self.list_of_features = set(list_of_features)
        return None
    
    def fit(self, dataset, y=None):
        return None
    
    def transform(self, dataset, y=None):
        data = dataset.copy()
        
        def get_time_features(r):
            
            features = []
            if 'month' in self.list_of_features:
                features.append(("_month", str(datetime.date(r).month)))
            if 'weekday' in self.list_of_features:
                features.append(("_weekday", str(datetime.weekday(r))))
            if 'is_month_end' in self.list_of_features:
                features.append(("_is_month_end", '1' if calendar.monthrange(datetime.date(r).year,datetime.date(r).month)[1] == datetime.date(r).day else '0'))
            if 'is_month_start' in self.list_of_features:
                features.append(("_is_month_start", '1' if datetime.date(r).day == 1 else '0'))
            return tuple(features)
        try: 
            for i in self.time_features:
                list_of_features = [get_time_features(r) for r in data[i]]

                dd = defaultdict(list)

                for x in list_of_features:
                    for k, v in x:
                        dd[k].append(v)

                for k, v in dd.items():
                    data[i+k] = v

                # if hour is also choosen
                if 'hour' in self.list_of_features:
                    h = [datetime.time(r).hour for r in data[i]]
                    if sum(h) > 0:
                        data[i+"_hour"] = h
                        data[i+"_hour"] = data[i+"_hour"].apply(str)

            data.drop(self.time_features, axis=1, inplace=True)
        except:
            print("Make Time Feature can not be done!")
        
        return data
    
    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        if not self.time_features:
            self.time_features = data.select_dtypes(include=['datetime64[ns]']).columns
        
        return self.transform(data)
            

class Empty(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    
    def fit(self, dataset, y=None):
        return None
    
    def transform(self, dataset, y=None):
        return dataset
    
    def fit_transform(self, dataset, y=None):
        return self.transform(dataset)

def Supervised_Path(train_data, target_variable, ml_usecase=None,
                   test_data=None, categorical_features=[], numerical_features=[], 
                   time_features=[], features_to_drop=[],
                   imputation_type="simple imputer", numeric_imputation_strategy="mean", categorical_imputation_strategy='most frequent',
                   apply_zero_nearZero_variance=False,
                   apply_grouping=False, group_name=[], features_to_group_ListofList=[[]],
                   scale_data=False, scaling_method='zscore',
                   target_transformation=False, target_transformation_method='bc',
                   Power_transform_data=False, Power_transform_method='quantile',
                   random_state=42):
    
    global subcase
    
    train_data.columns = [str(i) for i in train_data.columns]
    if test_data is not None:
        test_data.columns = [str(i) for i in test_data.columns]
        
    c1 = train_data[target_variable].dtype == 'int64'
    c2 = train_data[target_variable].nunique() <= 20
    c3 = train_data[target_variable].dtype.name in ['object', 'bool' 'category']
    
    if ml_usecase is None:
        if (c1 & c2) | c3:
            ml_usecase = 'classification'
        else:
            ml_usecase = 'regression'
            
    if (train_data[target_variable].nunique() > 2) and (ml_usecase != 'regression'):
        subcase = 'multi'
    else:
        subcase = 'binary'
        
    dtypes = Handle_Datatype(target=target_variable, ml_usecase=ml_usecase, categorical_features=categorical_features,
                            numerical_features=numerical_features, time_features=time_features, features_to_drop=features_to_drop)
    
    if imputation_type == 'simple imputer':
        try:
            imputer = Handle_Missing(target_variable=target_variable, numeric_strategy=numeric_imputation_strategy,
                                     categorical_strategy=categorical_imputation_strategy)
        except Exception as e:
            print(e)
    else:
        imputer = Empty()
    
    if apply_zero_nearZero_variance == True:
        znz = Handle_Zero_NearZero_Variance(target=target_variable)
    else:
        znz = Empty()
        
    if apply_grouping == True:
        group = Group_Similiar_Features(group_name=group_name, list_of_grouped_features=features_to_group_ListofList)
    else:
        group = Empty()
        
    if scale_data == True:
        scaling = Scaling_and_Power_Transformation(target=target_variable, function_to_apply=scaling_method,
                                                  random_state_quantile=random_state)
    else:
        scaling = Empty()
    
    if Power_transform_data == True:
        p_transform = Scaling_and_Power_Transformation(target=target_variable, function_to_apply=Power_transform_method, 
                                                      random_state_quantile=random_state)
    else:
        p_transform = Empty()
        
    if (target_transformation == True) and (ml_usecase == 'regression'):
        pt_target = Target_Transformation(target=target_variable, function_to_apply=target_transformation_method)
    else:
        pt_target = Empty()
        
    try:
        time_feature = Make_Time_Features()
    except:
        time_feature = Empty()
        
    pipe = Pipeline([
        ('dtypes', dtypes),
        ('imputer', imputer),
        ('znz', znz),
        ('group', group),
        ('scaling', scaling),
        ('p_transform', p_transform),
        ('pt_target', pt_target),
        ('time_feature', time_feature)
    ])
    
    if test_data is not None:
        return pipe.fit_transform(train_data), pipe.transform(test_data)
    else:
        return pipe.fit_transform(train_data)

