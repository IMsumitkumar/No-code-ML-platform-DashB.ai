import os
from django.shortcuts import render, redirect
from .models import DataSet
from .forms import DataSetForm
from .operations import *
from .user_code import *
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .data_prepration import *


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
            csv_df = df.head(20)
            obj = csv_df.to_html()

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
        'obj': obj,
        'data_desc_obj': data_desc.to_html(),
        'num_desc': num_null_desc,
        'cat_desc': cat_null_desc,
        'corr': correlation.to_html(),

    }

    return render(request, 'data/boot.html', context)


@login_required()
def Operation(request):
    global code_operation
    operations = {}
    column_name = []
    data_operation = []
    code_snippet = []
    data = DataSet.objects.filter(user=request.user).order_by('-id')[:1]
    for i in data:
        data = pd.read_csv('media' + i.file.url, error_bad_lines=False, encoding='latin-1')
        if data.shape[0] > 100000:
            data = data[0:100000]
        if request.method == 'POST':
            target_column = request.POST.get("target-col") or None
            # zero_variance = request.POST.get("zero-variance") or None
            # cat_feature_with_rare_levels = request.POST.get("new-cat-feature-by-rare") or None
            # group_sim_features = request.POST.get("Gp-sim-feature") or None
            scale_n_transform = request.POST.get("scale") or None
            target_scale = request.POST.get("target-scale") or None
            # dim_reduction = request.POST.get("dim-reduction") or None

            zero_variance = True if request.POST.get("zero-variance")=='YES' else False
            cat_feature_with_rare_levels = True if request.POST.get("new-cat-feature-by-rare") == 'YES' else False
            group_sim_features = True if request.POST.get("Gp-sim-feature") == 'YES' else False
            dim_reduction = True if request.POST.get("dim-reduction") == 'YES' else False

            print(scale_n_transform)

            for col in data.columns:
                column_name.append(col)
                data_operation.append(request.POST.get(col) or None)
                code_snippet.append(request.POST.get("code-" + col))

            operations = dict(zip(column_name, data_operation))
            code_operation = dict(zip(column_name, code_snippet))

            for operation in operations:
                if operations[operation] == "Int64":
                    intConversion(request, data, operation)
                elif operations[operation] == 'Float64':
                    floatConversion(request, data, operation)
                elif operations[operation] == "Drop":
                    drop_column(data, operation)
                elif operations[operation] == "DropNanFromRows":
                    drop_nan_rows(data)
                elif operations[operation] == "DropNanFromColumn":
                    drop_nan_cols(data)
                elif operations[operation] == "DropDuplicate":
                    drop_duplicate(data, operation)
                elif operations[operation] == "FillNanWithZero":
                    fill_nan_with_zero_or_drop(request, data, operation)
                elif operations[operation] == "FillNanWithPrecedenceAndZero":
                    fill_nan_precedureAndZero_or_drop(request, data, operation)
                elif operations[operation] == "FillNanWithOccurance":
                    fill_nan_max_occurance_or_drop(request, data, operation)
                elif operations[operation] == "FillNanWithMean":
                    fill_nan_with_mean_or_drop(request, data, operation)
                elif operations[operation] == "FillNanWithMedian":
                    fill_nan_with_median_or_drop(request, data, operation)
                elif operations[operation] == "FillNanWithMode":
                    fill_nan_with_mode_or_drop(request, data, operation)
                elif operations[operation] == "FillNanWithStd":
                    fill_nan_with_std_or_drop(request, data, operation)
                elif operations[operation] == 'FillNanWithOccurance>LabelEncode':
                    fill_nan_max_occurance_or_drop_then_label_encoding(request, data, operation)
                elif operations[operation] == 'FillNanWithMean>LabelEncode':
                    fill_nan_with_mean_or_drop_then_label_encoding(request, data, operation)
                elif operations[operation] == 'FillNanWithMedian>LabelEncode':
                    fill_nan_with_median_or_drop_then_label_encoding(data, operation)
                elif operations[operation] == 'FillNanWithMode>LabelEncode':
                    fill_nan_with_mode_or_drop_then_label_encoding(data, operation)
                elif operations[operation] == 'FillNanWithOccurance>OneHotEncode':
                    data = fill_nan_max_occurance_or_drop_then_onehot_encoding(request, data, operation)
                elif operations[operation] == 'FillNanWithMean>OneHotEncode':
                    data = fill_nan_with_mean_or_drop_then_onehot_encoding(request, data, operation)
                elif operations[operation] == 'FillNanWithMedian>OneHotEncode':
                    data = fill_nan_with_median_or_drop_then_onehot_encoding(request, data, operation)
                elif operations[operation] == 'FillNanWithMode>OneHotEncode':
                    data = fill_nan_with_mode_or_drop_then_onehot_encoding(request, data, operation)
                else:
                    print("Not Supported")

            for col in code_operation:
                user_given(request, data, col, code_operation[col])


            data = Preprocess_Path_Supervised(train_data=data, target_variable=target_column, apply_zero_nearZero_variance=zero_variance, 
                                              apply_grouping=group_sim_features, scale_data=True, target_transformation=True)

            obj_data = data.head(20)
            obj = obj_data.to_html()

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
                'obj': obj,
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
