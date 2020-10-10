import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from django.contrib import messages


#############################################################################################

# return a dictionary having key=>Column_name and value=>Count of missing values of that specific coloumn
def isNull(df):
    null_col = []
    null_count = []
    for col in df.columns:
        total_null_len = len(df[col][df[col].isnull() == True])
        if total_null_len > 0:
            null_count.append(total_null_len)
            null_col.append(col)
    return dict(zip(null_col, null_count))


#################################################################################################

def intConversion(request, data, col):
    try:
        data[col] = data[col].apply(np.int64)
        return data
    except Exception as e:
        messages.error(request, 'Sorry! FillNanWithStd cannot be operated on this feature.')


def floatConversion(request, data, col):
    try:
        data[col] = data[col].apply(np.float64)
        return data
    except Exception as e:
        print("Oops!", e.__class__, "occurred.")
        messages.error(request, 'Sorry! FillNanWithStd cannot be operated on this feature.')


###################################################################################################

# OPERATION : "drop" => will drop the specific column
def drop_column(data, col):
    return data.drop(col, axis=1, inplace=True)


# OPERATION : "DropNanFromRows" => will Drop NA Value from rows
# It will drop all rows with atleast 1 NA Value
def drop_nan_rows(data):
    return data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)


# OPERATION : "DropNanFromColumn" => will Drop NA Value from column
# It will drop all column with atleast 1 NA Value
def drop_nan_cols(data):
    return data.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)


################################################################################################

def drop_duplicate(data, col):

    return data.drop(col, axis=1).drop_duplicates()



################################################################################################

def fill_nan_with_zero_or_drop(request, data, col):
    data_shape = data.shape
    column_null_count = isNull(pd.DataFrame(data[col]))
    if column_null_count[col] >= int(data_shape[0] / 2):
        drop_column(data, col)
    else:
        data[col].fillna(0)
        if data[col].dtypes != 'object':
            data[col] = data[col].apply(np.int64)
        else:
            messages.error(request, 'Can not convert string into Integer.')
    return data


def fill_nan_precedureAndZero_or_drop(request, data, col):
    data_shape = data.shape
    column_null_count = isNull(pd.DataFrame(data[col]))
    if column_null_count[col] >= int(data_shape[0] / 2):
        drop_column(data, col)
    else:
        data[col].fillna(method='bfill', axis=0).fillna(0)
        if data[col].dtypes != 'object':
            data[col] = data[col].apply(np.int64)
        else:
            messages.error(request, 'Can not convert string into Integer.')
    return data


def fill_nan_max_occurance_or_drop(request, data, col):
    data_shape = data.shape
    column_null_count = isNull(pd.DataFrame(data[col]))
    try:
        if column_null_count[col] >= int(data_shape[0] / 2):
            drop_column(data, col)
        else:
            data[col].fillna(data[col].value_counts().index[0])
            if data[col].dtypes != 'object':
                data[col] = data[col].apply(np.int64)
            else:
                messages.error(request, 'Can not convert string into Integer.')
    except Exception as e:
        messages.error(request, 'Can not convert string into Integer.')
    return data


def fill_nan_with_occurance(data):
    return data.apply(lambda x: x.fillna(x.value_counts().index[0]))


def fill_nan_with_mean_or_drop(request, data, col):
    print(data[col].dtypes)
    data_shape = data.shape
    column_null_count = isNull(pd.DataFrame(data[col]))
    if column_null_count[col] >= int(data_shape[0] / 2):
        drop_column(data, col)
    else:
        data[col].fillna(value=data[col].mean(), inplace=True)
        if data[col].dtypes != 'object':
            data[col] = data[col].apply(np.int64)
        else:
            messages.error(request, 'Can not convert string into Integer.')
    return data


def fill_nan_with_median_or_drop(request, data, col):
    data_shape = data.shape
    column_null_count = isNull(pd.DataFrame(data[col]))
    if column_null_count[col] >= int(data_shape[0] / 2):
        drop_column(data, col)
    else:
        data[col].fillna(value=data[col].median(), inplace=True)
        if data[col].dtypes != 'object':
            data[col] = data[col].apply(np.int64)
        else:
            messages.error(request, 'Can not convert string into Integer.')
    return data


def fill_nan_with_mode_or_drop(request, data, col):
    data_shape = data.shape
    column_null_count = isNull(pd.DataFrame(data[col]))
    if column_null_count[col] >= int(data_shape[0] / 2):
        drop_column(data, col)
    else:
        data[col].fillna(value=data[col].mode()[0], inplace=True)
        if data[col].dtypes != 'object':
            data[col] = data[col].apply(np.int64)
        else:
            messages.error(request, 'Can not convert string into Integer.')
    return data


def fill_nan_with_std_or_drop(request, data, col):
    data_shape = data.shape
    column_null_count = isNull(pd.DataFrame(data[col]))
    if column_null_count[col] >= int(data_shape[0] / 2):
        drop_column(data, col)
    else:
        try:
            data[col].fillna(value=data[col].std(), inplace=True)
            data[col] = data[col].apply(np.int64)
        except Exception as e:
            messages.error(request, 'Sorry! FillNanWithStd cannot be operated on this feature.')
    return data


#################################################################################################### sklearn.LabelEncode

def fill_nan_max_occurance_or_drop_then_label_encoding(request, fill_nan_max_occurance_or_drop, col):
    data = fill_nan_max_occurance_or_drop
    labelencoder = LabelEncoder()
    if data[col].dtypes == 'object':
        try:
            data[col] = labelencoder.fit_transform(data[col].astype(str))
        except Exception as e:
            messages.error(request, 'Sorry! FillNanWithStd cannot be operated on this feature.')
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data


def fill_nan_with_mean_or_drop_then_label_encoding(request, fill_nan_with_mean_or_drop, col):
    data = fill_nan_with_mean_or_drop
    labelencoder = LabelEncoder()
    if data[col].dtypes == 'object':
        try:
            data[col] = labelencoder.fit_transform(data[col].astype(str))
        except Exception as e:
            messages.error(request, 'Sorry! FillNanWithStd cannot be operated on this feature.')
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data


def fill_nan_with_median_or_drop_then_label_encoding(fill_nan_with_median_or_drop, col):
    data = fill_nan_with_median_or_drop
    labelencoder = LabelEncoder()
    if data[col].dtypes == 'object':
        try:
            data[col] = labelencoder.fit_transform(data[col].astype(str))
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print("THIS OPERATION CAN NOT BE PERFORMED ON STRING DATA")
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data


def fill_nan_with_mode_or_drop_then_label_encoding(fill_nan_with_mode_or_drop, col):
    data = fill_nan_with_mode_or_drop
    labelencoder = LabelEncoder()
    if data[col].dtypes == 'object':
        try:
            data[col] = labelencoder.fit_transform(data[col].astype(str))
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print("THIS OPERATION CAN NOT BE PERFORMED ON STRING DATA")
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data


############################################################################################################ sklearn.LabelBinarizer

def fill_nan_max_occurance_or_drop_then_onehot_encoding(request, fill_nan_max_occurance_or_drop, col):
    data = fill_nan_max_occurance_or_drop
    if data[col].dtypes == 'object':
        try:
            if data.shape[0] <= 10000:
                onehotencoder = LabelBinarizer()
                hot_df = onehotencoder.fit_transform(data[col].astype(str))
                result_df = pd.DataFrame(hot_df, columns=onehotencoder.classes_)
                data = pd.concat([data, result_df], axis=1)
                # Task: Drop the column which is being hot encoded
            else:
                messages.error(request, "Columns limit reached")
        except Exception as e:
            messages.error(request, "Not Operable")
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data


def fill_nan_with_mean_or_drop_then_onehot_encoding(request, fill_nan_with_mean_or_drop, col):
    data = fill_nan_with_mean_or_drop
    if data[col].dtypes == 'object':
        try:
            if data.shape[0] <= 10000:
                onehotencoder = LabelBinarizer()
                hot_df = onehotencoder.fit_transform(data[col].astype(str))
                result_df = pd.DataFrame(hot_df, columns=onehotencoder.classes_)
                data = pd.concat([data, result_df], axis=1)
                # Task: Drop the column which is being hot encoded
            else:
                messages.error(request, "Columns limit reached")
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print("THIS OPERATION CAN NOT BE PERFORMED ON STRING DATA --> This is one hot encoding")
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data


def fill_nan_with_median_or_drop_then_onehot_encoding(request, fill_nan_with_median_or_drop, col):
    data = fill_nan_with_median_or_drop
    if data[col].dtypes == 'object':
        try:
            if data.shape[0] <= 10000:
                onehotencoder = LabelBinarizer()
                hot_df = onehotencoder.fit_transform(data[col].astype(str))
                result_df = pd.DataFrame(hot_df, columns=onehotencoder.classes_)
                data = pd.concat([data, result_df], axis=1)
                # Task: Drop the column which is being hot encoded
            else:
                messages.error(request, "Columns limit reached")
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print("THIS OPERATION CAN NOT BE PERFORMED ON STRING DATA --> This is one hot encoding")
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data


def fill_nan_with_mode_or_drop_then_onehot_encoding(request, fill_nan_with_mode_or_drop, col):
    data = fill_nan_with_mode_or_drop
    if data[col].dtypes == 'object':
        try:
            if data.shape[0] <= 10000:
                onehotencoder = LabelBinarizer()
                hot_df = onehotencoder.fit_transform(data[col].astype(str))
                result_df = pd.DataFrame(hot_df, columns=onehotencoder.classes_)
                data = pd.concat([data, result_df], axis=1)
                # Task: Drop the column which is being hot encoded
            else:
                messages.error(request, "Columns limit reached")
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
            print("THIS OPERATION CAN NOT BE PERFORMED ON STRING DATA --> This is one hot encoding")
    else:
        print("TERI MAA KI....BHAI CHOOSE CORRECT COLUMN")
    return data

# Task:
# 1. remove column if column have more than 50% unique values


################################################################# Outliers