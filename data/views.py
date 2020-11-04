import os
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from django.shortcuts import render, redirect, resolve_url
from django.http import HttpResponse
from .models import DataSet, ProcessedDataSet
from .forms import DataSetForm, ProcessedDatasetForm
from .user_code import *
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .preprocess import Supervised_Path
import json2html
from .get_db_data import *
from django.conf import settings 

@login_required()
def upload_data(request):
    obj = None

    if request.method == 'POST':
        form = DataSetForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.user = request.user

            if len(request.FILES) != 0:
                _, ext = os.path.splitext(instance.file.url)
                if ext != '.csv':
                    messages.error(request, "UnSupported File format. Please Upload .csv data :)")
                    return redirect('upload')
                else:
                    instance.save()
                    return redirect('dashboard')
            else:
                messages.error(request, "You haven't upload any dataset :(")


    else:
        form = DataSetForm()

    return render(request, 'data/index.html', {'form': form})


@login_required()
def dashBoard(request):

    if DataSet.objects.filter(user=request.user).count() == 0:
        messages.error(request, "Please upload the data first.")
        return redirect('upload')
    else:
        data = DataSet.objects.filter(user=request.user).order_by('-id')[:1]
        for i in data:
            df = pd.read_csv('media' + i.file.url, error_bad_lines=False, encoding='latin-1')

    
    csv_df_top, csv_df_bottom = df.head(), df.tail()

    data_shape = df.shape
    data_columns = df.columns
    data_desc = df.describe()

    categorical_data = df.dtypes[df.dtypes == 'O'].index
    num1 = df.dtypes[df.dtypes == 'int64'].index
    num2 = df.dtypes[df.dtypes == 'float64'].index
    numerical_data = num1.append(num2)

    cat_null_desc = datatype_nullCount(df, categorical_data)
    num_null_desc = datatype_nullCount(df, numerical_data)

    # matrix = np.triu(df.corr())
    # correlation = sns.heatmap(df.corr(), annot=True, mask=matrix)
    try:
        correlation = px.imshow(df.corr(), width=1000, height=500)
    except:
        correlation = df.corr()

    context = {
        'rows': data_shape[0],
        'columns': data_shape[1],
        'data_columns': data_columns,
        'obj_top': csv_df_top.to_html(),
        'obj_bottom':csv_df_bottom.to_html(),
        'data_desc_obj': data_desc.to_html(),
        'num_desc': num_null_desc,
        'cat_desc': cat_null_desc,
        'corr': correlation.to_html(),
        'cat_column':categorical_data,
    }

    return render(request, 'data/boot.html', context)


@login_required()
def Operation(request):
    operations = {}
    column_name = []
    data_operation = []
    code_snippet = []
    empty_list = []
    data = DataSet.objects.filter(user=request.user).order_by('-id')[:1]
    for i in data:
        data = pd.read_csv('media' + i.file.url, error_bad_lines=False, encoding='latin-1')
        if data.shape[0] > 100000:
            data = data[0:100000]
        # if request.method == 'POST':
        target_variable = request.POST.get("target-col") or None
        time_features = request.POST.getlist("time-features") or empty_list
        features_to_drop = request.POST.getlist("toDrop-features") or empty_list
        numeric_imputation_strategy = request.POST.get("num-stats") or 'zscore'
        categoric_imputation_strategy = request.POST.get("cat-stats") or 'most frequent'
        remove_zero_variance = True if request.POST.get("rm-var") == 'yes' else False                  
        group_sim_features = True if request.POST.get("gp-sim-feature") == 'yes' else False     
        sim_group_name = request.POST.getlist("gp-sim-gp-name") or empty_list
        sim_feature_list = request.POST.getlist("gp-sim-feature-list") or empty_list
        scale_and_transform = True if request.POST.get("scale-data") == 'yes' else False             
        scale_and_transform_method = request.POST.get("scale-method") or None
        target_transform = True if request.POST.get("target-transform") == 'yes' else False            
        power_transform = True if request.POST.get("power-transform") == 'yes' else False 
        nominal = True if request.POST.get("nominal") == 'yes' else False  
        ordinal = True if request.POST.get("ordinal") == 'yes' else False  
        nominal_method = request.POST.get("nominal-method") or 'kdd orange'
        ordinal_method = request.POST.get("ordinal-method") or 'target guided'
        nominal_features = request.POST.getlist("nominal-features") or empty_list
        ordinal_features = request.POST.getlist("ordinal-features") or empty_list
        top_features_for_kdd = request.POST.get("top-features") or 10

        apply_outlier = True if request.POST.get("remove-outlier") == 'yes' else False
        outlier_method = request.POST.getlist("remove-outlier-method") or 'iso'
        apply_feature_selection = True if request.POST.get("feature-selection") == 'yes' else False
        feature_selection_method = request.POST.get("feature-selection-methods") or 'lgbm'
        limit_features = request.POST.get("limit-features") or 10

        # try:
        data = Supervised_Path(train_data=data, target_variable=target_variable,
                            time_features=time_features, features_to_drop=features_to_drop,numeric_imputation_strategy=numeric_imputation_strategy,
                            categorical_imputation_strategy=categoric_imputation_strategy,
                            apply_zero_nearZero_variance=remove_zero_variance,
                            apply_grouping=group_sim_features, group_name=sim_group_name, features_to_group_ListofList=[sim_feature_list],
                            nominal_encoding=nominal, top=int(top_features_for_kdd), nominal_encoding_method =nominal_method, features_for_nominal_encode=nominal_features,
                            ordinal_encoding=ordinal, ordinal_encoding_method =ordinal_method, features_for_ordinal_encode=ordinal_features,
                            scale_data=scale_and_transform, scaling_method=scale_and_transform_method,
                            remove_outliers=apply_outlier, outlier_methods=outlier_method,
                            apply_feature_selection=apply_feature_selection, feature_selection_method=feature_selection_method, limit_features=int(limit_features),
                            target_transformation=target_transform, Power_transform_data=power_transform)
        # except Exception as e: 
        #     messages.error(request, "Target column is not selected"+str(e))


        data.to_csv(settings.MEDIA_ROOT / 'output' / 'output.csv')

        # DataSet.objects.create(user=request.user, file=data.to_csv(settings.MEDIA_ROOT / 'output' / 'boutput.csv'))

        obj_top, obj_bottom = data.head(), data.tail()

        # categorical_data = data.select_dtypes(include=['object']).columns
        # numerical_data = data.select_dtypes(include=['int64', 'float64']).columns

        categorical_data = data.dtypes[data.dtypes == 'O'].index
        num1 = data.dtypes[data.dtypes == 'int64'].index
        num2 = data.dtypes[data.dtypes == 'float64'].index
        numerical_data = num1.append(num2)

        data_shape, data_columns, null_count, data_desc = about_data(data)
        cat_null_desc = datatype_nullCount(data, categorical_data)
        num_null_desc = datatype_nullCount(data, numerical_data)

        context = {
            'rows': data_shape[0],
            'columns': data_shape[1],
            'data_columns': data_columns,
            'obj_top': obj_top.to_html(),
            'obj_bottom':obj_bottom.to_html(),
            'data_desc_obj': data_desc.to_html(),
            'null_count': null_count,
            'num_desc': num_null_desc,
            'cat_desc': cat_null_desc,
        }
        return render(request, 'data/boot.html', context)


def about_data(data):
    data_shape = data.shape
    data_columns = data.columns
    null_count = data.isnull().sum()
    data_desc = data.describe()
    return data_shape, data_columns, null_count, data_desc


def datatype_nullCount(df, indexes):
    null_count = []
    for i in df[indexes].isnull().sum():
        null_count.append(i)
    return dict(zip(indexes, null_count))


def save_processed_dataset(request):
    # data = request.session.get('processed_data')

    # x = json2html.json2html.convert(json=data)
    context = {}
    return render(request, 'data/modeling.html', context)


def database_dash(request):
    db_host = None

    if request.method == 'POST':
        db_host = request.POST.get("db-host") or None
        db_username = request.POST.get("db-username") or None
        db_password = request.POST.get("db-password") or None
        db_name = request.POST.get("db-database") or None
        db_table = request.POST.get("db-tablename") or None

    if (db_host and db_username) is not None:
        try:
            data = DB_from_servers(connect_to_mysql=True, host=db_host, username=db_username, password=db_password, 
                                database=db_name, table_name=db_table)
            obj_top, obj_bottom = data.head(), data.tail()
            context = {
                'obj_top':obj_top.to_html(), 
                'obj_bottom':obj_bottom.to_html(),
            }
            return render(request, 'data/boot.html', context)
        except Exception as e:
            messages.error(request, "Install MySql server in your system | or " + str(e))
            return render(request, 'data/index.html', {})
    else:
        messages.error(request, "Enter Valid Credentials!")
        return render(request, 'data/index.html', {})
    