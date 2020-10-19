from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data.models import DataSet
from .plots import *
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .plots import graph

@login_required()
def visual(request):

    if DataSet.objects.filter(user=request.user).count() == 0:
        messages.error(request, "Please upload the data first.")
        return redirect('upload')
    else:
        data = DataSet.objects.filter(user=request.user).order_by('-id')[:1]
        for i in data:
            df = pd.read_csv('media' + i.file.url, error_bad_lines=False, encoding='latin-1')

    data_shape = df.shape
    data_columns = df.columns
    data_desc = df.describe()

    categorical_data = df.dtypes[df.dtypes == 'O'].index
    num1 = df.dtypes[df.dtypes == 'int64'].index
    num2 = df.dtypes[df.dtypes == 'float64'].index
    numerical_data = num1.append(num2)

    app = DjangoDash('SimpleExample', 
                serve_locally=True,
                add_bootstrap_links=True,
                meta_tags=[{'name':'viewport',
                'content':'width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no'}])   # replaces dash.Dash

    graph(app=app, df=df)

    context = {
        'data_columns': data_columns,
        'categorical_data': categorical_data,
        'numerical_data': numerical_data,
    }

    return render(request, 'viz/visual.html', context)


