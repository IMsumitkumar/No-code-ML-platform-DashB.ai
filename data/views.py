import os
import pandas as pd
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
    global obj

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
        if request.method == 'POST':
            target_variable = request.POST.get("target-col") or None
            time_features = request.POST.getlist("time-features") or empty_list
            features_to_drop = request.POST.getlist("toDrop-features") or empty_list
            numeric_imputation_strategy = request.POST.get("num-stats") or None
            categoric_imputation_strategy = request.POST.get("cat-stats") or None
            remove_zero_variance = True if request.POST.get("rm-var") == 'YES' else False                  
            group_sim_features = False if request.POST.get("gp-sim-feature") == 'YES' else True      
            sim_group_name = request.POST.getlist("gp-sim-gp-name") or empty_list
            sim_feature_list = request.POST.getlist("gp-sim-feature-list") or empty_list
            scale_and_transform = False if request.POST.get("scale") == 'YES' else True                   
            scale_and_transform_method = request.POST.get("scale-method") or None
            target_transform = False if request.POST.get("target-transform") == 'YES' else True            
            power_transform = False if request.POST.get("power-transform") == 'YES' else True 
                   

            try:
                data = Supervised_Path(train_data=data, target_variable=target_variable,
                                    time_features=time_features, features_to_drop=features_to_drop,numeric_imputation_strategy=numeric_imputation_strategy,
                                    categorical_imputation_strategy=categoric_imputation_strategy, apply_zero_nearZero_variance=remove_zero_variance,
                                    apply_grouping=False, group_name=[], features_to_group_ListofList=[[]],
                                    scale_data=scale_and_transform, scaling_method=scale_and_transform_method,
                                    target_transformation=target_transform, Power_transform_data=power_transform)
            except Exception as e:
                messages.error(request, "Columns are is override, Data can not be processed! OR you must roeload this page  "+ str(e))



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
    