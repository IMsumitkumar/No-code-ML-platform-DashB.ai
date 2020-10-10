import sys
import calendar
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from collections import defaultdict

class Auto_Datatypes(BaseEstimator, TransformerMixin):

    def __init__(self, target, ml_usecase, categorical_features=[], numerical_features=[], time_features=[],
                 features_todrop=[], display_types=True):

        self.replacement = {}
        self.id_columns = []
        self.target = target
        self.ml_usecase = ml_usecase
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.time_features = time_features
        self.features_todrop = features_todrop
        self.display_types = display_types

    def fit(self, dataset):
        # make dataset copy to work on
        data = dataset.copy()
        # drop all features which are asked to drop
        data.drop(columns=self.features_todrop, errors='ignore', inplace=True)
        # replace inf +/- values with NaN
        data.replace([np.inf, -np.inf], np.NaN, inplace=True)
        # try to convert categorical data into integer if possible
        for i in data.select_dtypes(include=['object']).columns:
            try:
                data[i] = data[i].astype('int64')
            except:
                None
        
        # convert pandas bool and category into categorical data 
        for i in data.select_dtypes(include=['bool', 'category']).columns:
            data[i] = data[i].astype('object')
        
        # try to find out the id column and remove it if not given by user to drop
        len_samples = len(data)
        for i in data.select_dtypes(include=['int64', 'float64']).columns:
            if i not in self.numerical_features:
                if sum(data[i].isna()) == 0:
                    if data[i].nunique() == len_samples:
                        features = data[i].sort_values()
                        increments = features.diff()[1:]
                        if sum(np.abs(increments - 1) < 1e-7) == len_samples - 1:
                            self.id_columns.append(i)
        
        # in pandas while working with csv files, if there is null value inside a int type column
        # pandas read it as float, find that and convert it
        for i in data.select_dtypes(include=['float64']).columns:
            na_count = sum(data[i].isna())
            count_float = np.nansum([False if r.is_integer() else True for r in data[i]])
            count_float = count_float - na_count
            if (count_float == 0) & (data[i].nunique() <= 20) & (na_count > 0):
                data[i] = data[i].astype('object')

                
        # Handling datetime features
        for i in data.select_dtypes(include=['object']).drop(self.target, axis=1, errors='ignore').columns:
            try:
                data[i] = pd.to_datetime(data[i], infer_datetime_format=True, utc=False, errors='raise')
            except:
                continue

        # now in case we were given any specific columns dtypes in advance
        # we will over ride those features
        for i in self.categorical_features:
            try:
                data[i] = data[i].apply(str)
            except:
                data[i] = dataset[i].apply(str)

        for i in self.numerical_features:
            try:
                data[i] = data[i].astype('float64')
            except:
                data[i] = dataset[i].astype('float64')

        for i in self.time_features:
            try:
                data[i] = pd.to_datetime(data[i], infer_datetime_format=True, utc=False, errors='raise')
            except:
                data[i] = pd.to_datetime(dataset[i], infer_datetime_format=True, utc=False, errors='raise')

        for i in data.select_dtypes(include=['datetime64']).columns:
            data[i] = data[i].astype('datetime64[ns]')

        # table of learent types
        self.learent_dtypes = data.dtypes
        # if there are inf or -inf then replace them with NaN
        data = data.replace([np.inf, -np.inf], np.NaN).astype(self.learent_dtypes)
        # lets remove duplicates
        data = data.loc[:, ~data.columns.duplicated()]
        # Remove NAs
        data.dropna(axis=0, how='all', inplace=True)
        data.dropna(axis=1, how='all', inplace=True)
        # remove the row if target column has NA
        data = data[~data[self.target].isnull()]

        # drop id columns now
        data.drop(self.id_columns, axis=1, errors='ignore', inplace=True)

        return data

    def transform(self, dataset, y=None):
        data = dataset.copy()

        # drop any columns that were asked to drop
        data.drop(columns=self.features_todrop, errors='ignore', inplace=True)
        # also make sure that all the column names are string
        data.columns = [str(i) for i in data.columns]
        # if there are inf or -inf then replace them with NaN
        data.replace([np.inf, -np.inf], np.NaN, inplace=True)
        for i in self.final_training_columns:
            if i not in data.columns:
                sys.exit('(Type Error): test data does not have column ' + str(i) + " which was used for training")

        # we only need to take test columns that we used in ttaining 
        # (test in production may have a lot more columns)
        data = data[self.final_training_columns]

        # just keep picking the data and keep applying to the test data set (be mindful of target variable)
          # we are taking all the columns in test , so we dot have to worry about droping target learnt col
        for i in data.columns:
            if self.learent_dtypes[i].name == 'datetime64[ns]':
                data[i] = pd.to_datetime(data[i], infer_datetime_format=True, utc=False, errors='coerce')
            data[i] = data[i].astype(self.learent_dtypes[i])

        # drop id columns
        data.drop(self.id_columns, axis=1, errors='ignore', inplace=True)

        return data

    def fit_transform(self, dataset, y=None):

        data = dataset.copy()
        # drop any columns that were asked to drop
        data.drop(columns=self.features_todrop, errors='ignore', inplace=True)

        # since this is for training , we dont nees any transformation since it has already been transformed in fit
        data = self.fit(data)

        # additionally we just need to treat the target variable
        # for ml use ase
        if (self.ml_usecase == 'classification') & (data[self.target].dtype == 'object'):
            le = LabelEncoder()
            data[self.target] = le.fit_transform(np.array(data[self.target]))

            # now get the replacement dict
            rev = le.inverse_transform(range(0, len(le.classes_)))
            rep = np.array(range(0, len(le.classes_)))
            for i, k in zip(rev, rep):
                self.replacement[i] = k

        # drop id columns
        data.drop(self.id_columns, axis=1, errors='ignore', inplace=True)
        # finally save a list of columns that we would need from test data set
        self.final_training_columns = data.drop(self.target, axis=1).columns

        return data

class Missing_Imputation(BaseEstimator, TransformerMixin):

    def __init__(self, numeric_strategy, categorical_strategy, target_variable):
        self.numeric_strategy = numeric_strategy
        self.target = target_variable
        self.categorical_strategy = categorical_strategy

    def fit(self, dataset, y=None):  #
        def zeros(x):
            return 0

        data = dataset.copy()
        # make a table for numerical variable with strategy stats
        if self.numeric_strategy == 'mean':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['float64', 'int64']).apply(
                np.nanmean)
        elif self.numeric_strategy == 'median':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['float64', 'int64']).apply(
                np.nanmedian)
        elif self.numeric_strategy == 'std':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['float64', 'int64']).apply(
                np.nanstd)
        elif self.numeric_strategy == 'zero':
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['float64', 'int64']).apply(
                zeros)
        else:
            self.numeric_stats = data.drop(self.target, axis=1).select_dtypes(include=['float64', 'int64']).apply(zeros)

        self.numeric_columns = data.drop(self.target, axis=1).select_dtypes(include=['float64', 'int64']).columns

        # for Catgorical ,
        if self.categorical_strategy == 'most frequent':
            self.categorical_columns = data.drop(self.target, axis=1).select_dtypes(include=['object']).columns
            self.categorical_stats = pd.DataFrame(columns=self.categorical_columns)  # place holder
            for i in self.categorical_stats.columns:
                self.categorical_stats.loc[0, i] = data[i].value_counts().index[0]
        else:
            self.categorical_columns = data.drop(self.target, axis=1).select_dtypes(include=['object']).columns

        # for time, there is only one way, pick up the most frequent one
        self.time_columns = data.drop(self.target, axis=1).select_dtypes(include=['datetime64[ns]']).columns
        self.time_stats = pd.DataFrame(columns=self.time_columns)  # place holder
        for i in self.time_columns:
            self.time_stats.loc[0, i] = data[i].value_counts().index[0]
        return data

    def transform(self, dataset, y=None):
        data = dataset.copy()
        # for numeric columns
        for i, s in zip(data[self.numeric_columns].columns, self.numeric_stats):
            data[i].fillna(s, inplace=True)

        # for categorical columns
        if self.categorical_strategy == 'most frequent':
            for i in self.categorical_stats.columns:

                data[i] = data[i].fillna(self.categorical_stats.loc[0, i])
                data[i] = data[i].apply(str)
        else:  # this means replace na with "not_available"
            for i in self.categorical_columns:
                data[i].fillna("not_available", inplace=True)
                data[i] = data[i].apply(str)
        # for time
        for i in self.time_stats.columns:
            data[i].fillna(self.time_stats.loc[0, i], inplace=True)

        return data

    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        data = self.fit(data)
        return self.transform(data)

class Zero_NearZero_Variance(BaseEstimator, TransformerMixin):
    '''
        - it eliminates the features having zero variance and non zero variance
        - Near zero variance is determined by 
            -1) Count of unique points divided by the total length of the feature has to be lower than a pre sepcified threshold 
            -2) Most common point(count) divided by the second most common point(count) in the feature is greater than a pre specified threshold
        Once both conditions are met , the feature is dropped  
    '''

    def __init__(self, target, threshold_1=0.1, threshold_2=20):
        self.to_drop = []
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.target = target

    def fit(self, dataset, y=None):  # from training data set we are going to learn what columns to drop
        data = dataset.copy()
        self.sampl_len = len(data[self.target])
        for i in data.drop(self.target, axis=1).columns:
            # get the number of unique counts
            u = pd.DataFrame(data[i].value_counts()).sort_values(by=i, ascending=False, inplace=False)
            # take len of u and divided it by the total sample numbers, so this will check the 1st rule , has to be low say 10%
            # import pdb; pdb.set_trace()
            first = len(u) / self.sampl_len
            # then check if most common divided by 2nd most common ratio is 20 or more
            if len(u[i]) == 1:  # this means that if column is non variance , automatically make the number big to drop it
                second = 100
            else:
                second = u.iloc[0, 0] / u.iloc[1, 0]
            # if both conditions are true then drop the column, however, we dont want to alter column that indicate NA's
            if (first <= 0.10) and (second >= 20) and (i[-10:] != '_surrogate'):
                self.to_drop.append(i)
                # now drop if the column has zero variance
            if (second == 100) and (i[-10:] != '_surrogate'):
                self.to_drop.append(i)

    def transform(self, dataset, y=None):  # since it is only for training data set , nothing here
        data = dataset.copy()
        data.drop(self.to_drop, axis=1, inplace=True)
        return data

    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        self.fit(data)
        return self.transform(data)

class Group_Similar_Features(BaseEstimator, TransformerMixin):

    def __init__(self, group_name=[], list_of_grouped_features=[[]]):
        self.list_of_similar_features = list_of_grouped_features
        self.group_name = group_name
        # if list of list not given
        try:
            np.array(self.list_of_similar_features).shape[0]
        except:
            raise ("Group_Similar_Features: list_of_grouped_features is not provided as list of list")

    def fit(self, data, y=None):
        return None

    def transform(self, dataset, y=None):
        data = dataset.copy()
        # only going to process if there is an actual missing value in training data set
        if len(self.list_of_similar_features) > 0:
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

    def fit_transform(self, data, y=None):
        self.fit(data)
        return self.transform(data)

class Binning(BaseEstimator, TransformerMixin):

    def __init__(self, features_to_discretize):
        self.features_to_discretize = features_to_discretize

    def fit(self, data, y=None):
        return None

    def transform(self, dataset, y=None):
        data = dataset.copy()
        # only do if features are provided
        if len(self.features_to_discretize) > 0:
            data_t = self.disc.transform(np.array(data[self.features_to_discretize]).reshape(-1, self.len_columns))
            # make pandas data frame
            data_t = pd.DataFrame(data_t, columns=self.features_to_discretize, index=data.index)
            # all these columns are catagorical
            data_t = data_t.astype(str)
            # drop original columns
            data.drop(self.features_to_discretize, axis=1, inplace=True)
            # add newly created columns
            data = pd.concat((data, data_t), axis=1)
        return data

    def fit_transform(self, dataset, y=None):
        data = dataset.copy()

        # only if features are given
        if len(self.features_to_discretize) > 0:

            # place holder for all the features for their binns
            self.binns = []
            for i in self.features_to_discretize:
                # get numbr of binns
                hist, bin_edg = np.histogram(data[i], bins='sturges')
                self.binns.append(len(hist))

            # how many colums to deal with
            self.len_columns = len(self.features_to_discretize)
            # now do fit transform
            self.disc = KBinsDiscretizer(n_bins=self.binns, encode='ordinal', strategy='kmeans')
            data_t = self.disc.fit_transform(np.array(data[self.features_to_discretize]).reshape(-1, self.len_columns))
            # make pandas data frame
            data_t = pd.DataFrame(data_t, columns=self.features_to_discretize, index=data.index)
            # all these columns are categorical
            data_t = data_t.astype(str)
            # drop original columns
            data.drop(self.features_to_discretize, axis=1, inplace=True)
            # add newly created columns
            data = pd.concat((data, data_t), axis=1)

        return data

class Scaling_and_Power_transformation(BaseEstimator, TransformerMixin):

    def __init__(self, target, function_to_apply='zscore', random_state_quantile=42):
        self.target = target
        self.function_to_apply = function_to_apply
        self.random_state_quantile = random_state_quantile

    def fit(self, dataset, y=None):

        data = dataset.copy()
        # we only want to apply if there are numeric columns
        self.numeric_features = data.drop(self.target, axis=1, errors='ignore').select_dtypes(
            include=["float64", 'int64']).columns
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
            return None

    def transform(self, dataset, y=None):
        data = dataset.copy()

        if len(self.numeric_features) > 0:
            self.data_t = pd.DataFrame(self.scale_and_power.transform(data[self.numeric_features]))
            # we need to set the same index as original data
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
        # if target has zero or negative values use yj auto
        if any(data[self.target] <= 0):
            self.function_to_apply = 'yeo-johnson'
        self.p_transform_target = PowerTransformer(method=self.function_to_apply)
        data[self.target] = self.p_transform_target.fit_transform(np.array(data[self.target]).reshape(-1, 1))

        return data

class Ordinal(BaseEstimator, TransformerMixin):

    def __init__(self, info_as_dict):
        self.info_as_dict = info_as_dict

    def fit(self, data, y=None):
        return None

    def transform(self, dataset, y=None):
        data = dataset.copy()
        new_data_test = pd.DataFrame(self.enc.transform(data[self.info_as_dict.keys()]),
                                     columns=self.info_as_dict.keys(), index=data.index)
        for i in self.info_as_dict.keys():
            data[i] = new_data_test[i]
        return data

    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        # creat categories from given keys in the data set
        cat_list = []
        for i in self.info_as_dict.values():
            i = [np.array(i)]
            cat_list = cat_list + i

        # now do fit transform
        self.enc = OrdinalEncoder(cat_list)
        new_data_train = pd.DataFrame(self.enc.fit_transform(data.loc[:, self.info_as_dict.keys()]),
                                      columns=self.info_as_dict, index=data.index)
        for i in self.info_as_dict.keys():
            data[i] = new_data_train[i]

        return data

class Clean_Colum_Names(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, data, y=None):
        return None

    def transform(self, dataset, y=None):
        data = dataset.copy()
        data.columns = data.columns.str.replace(r'[^A-Za-z0-9]+', '')
        return data

    def fit_transform(self, dataset, y=None):
        return self.transform(dataset, y=y)

class Empty(BaseEstimator, TransformerMixin):

    def __init__(self):
        return (None)

    def fit(self, data, y=None):
        return None

    def transform(self, data, y=None):
        return data

    def fit_transform(self, data, y=None):
        return self.transform(data)

def Preprocess_Path_Supervised(train_data, target_variable, ml_usecase=None, test_data=None, categorical_features=[],
                        numerical_features=[], time_features=[],display_types=True,
                        imputation_type="simple imputer",features_to_drop=[], numeric_imputation_strategy='mean',
                        categorical_imputation_strategy='most frequent',
                        apply_zero_nearZero_variance=False,
                        apply_binning=False, features_to_binn=[],
                        apply_grouping=False, group_name=[], features_to_group_ListofList=[[]],
                        scale_data=False, scaling_method='zscore',
                        target_transformation=False, target_transformation_method='bc',
                        random_state=42):

    global subcase

    train_data.columns = [str(i) for i in train_data.columns]
    if test_data is not None:
        test_data.columns = [str(i) for i in test_data.columns]

    c1 = train_data[target_variable].dtype == 'int64'
    c2 = train_data[target_variable].nunique() <= 20
    c3 = train_data[target_variable].dtype.name in ['object', 'bool', 'category']

    if ml_usecase is None:
        if (c1 & c2) | c3:
            ml_usecase = 'classification'
        else:
            ml_usecase = 'regression'

    if (train_data[target_variable].nunique() > 2) and (ml_usecase != 'regression'):
        subcase = 'multi'
    else:
        subcase = 'binary'

    global dtypes
    dtypes = Auto_Datatypes(target=target_variable, ml_usecase=ml_usecase,
                                  categorical_features=categorical_features, numerical_features=numerical_features,
                                  time_features=time_features, features_todrop=features_to_drop,
                            display_types=display_types)

    if imputation_type == "simple imputer":
        global imputer
        imputer = Missing_Imputation(numeric_strategy=numeric_imputation_strategy, target_variable=target_variable,
                                 categorical_strategy=categorical_imputation_strategy)
    else:
        imputer = Empty()

    if apply_zero_nearZero_variance == True:
        global znz
        znz = Zero_NearZero_Variance(target=target_variable)
    else:
        znz = Empty()




    if apply_grouping == True:
        global group
        group = Group_Similar_Features(group_name=group_name, list_of_grouped_features=features_to_group_ListofList)
    else:
        group = Empty()


    if apply_binning == True:
        global binn
        binn = Binning(features_to_discretize=features_to_binn)
    else:
        binn = Empty()


    if scale_data == True:
        global scaling
        scaling = Scaling_and_Power_transformation(target=target_variable, function_to_apply=scaling_method,
                                                   random_state_quantile=random_state)
    else:
        scaling = Empty()

    


    if (target_transformation == True) and (ml_usecase == 'regression'):
        global pt_target
        pt_target = Target_Transformation(target=target_variable, function_to_apply=target_transformation_method)
    else:
        pt_target = Empty()



    # clean column names for special char
    clean_names = Clean_Colum_Names()



    global pipe
    pipe = Pipeline([
        ('dtypes', dtypes),
        ('imputer', imputer),
        ('znz', znz),
        ('group', group),
        ('scaling', scaling),
        ('pt_target', pt_target),
        ('binn', binn),
        ('clean_names', clean_names),
    ])

    if test_data is not None:
        return pipe.fit_transform(train_data), pipe.transform(test_data)
    else:
        return pipe.fit_transform(train_data)

