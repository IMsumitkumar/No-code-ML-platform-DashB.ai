from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from data.models import DataSet
from .plots import *
from django.contrib import messages
from django.contrib.auth.decorators import login_required

@login_required()
def visual(request):
    global y_fig

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

    plots = ['Bar Plot', 'Scatter Plot', 'Line Plot', 'Pie Plot', 'Box Plot',
             'Box Plot (Algo)', 'Histogram Plot', 'Heat Map', 'Tree Map']

    if request.method == 'POST':
        plot_type = request.POST.get('plot-type')
        x_axis = request.POST.get('x-axis') or None
        y_axis = request.POST.get('y-axis') or None
        color = request.POST.get('color') or None
        size = request.POST.get('size') or None
        names = request.POST.get('names') or None
        title = request.POST.get('title') or None


        if plot_type == 'Bar Plot':
            y_fig = plot_bar(df, x_axis, y_axis, color)
        elif plot_type == 'Scatter Plot':
            y_fig = plot_scatter(df, x_axis, y_axis)
        elif plot_type == 'Line Plot':
            y_fig = plot_line(df, x_axis, y_axis)
        elif plot_type == 'Pie Plot':
            y_fig = plot_pie(df, x_axis, names)
        elif plot_type == 'Box Plot':
            y_fig = plot_box(df, x_axis, y_axis, color)
        elif plot_type == 'Box Plot (Algo)':
            y_fig = plot_box_algo(df, x_axis, y_axis)
        elif plot_type == 'Histogram Plot':
            y_fig = plot_histogram(df, x_axis)
        elif plot_type == 'Heat Map':
            y_fig = density_heat_map(df, x_axis, y_axis)
        elif plot_type == 'Tree Map':
            y_df = px.data.gapminder().query("year == 2007")
            y_fig = px.treemap(y_df, path=[px.Constant('world'), 'continent', 'country'], values='pop',
                               color='lifeExp', hover_data=['iso_alpha'])
        else:
            print('Not Supported !')

        context = {
            'plots': plots,
            'data_columns': data_columns,
            'y_fig': y_fig.to_html(),
            'categorical_data': categorical_data,
            'numerical_data': numerical_data,
        }

        return render(request, 'viz/visuals.html', context)

    y_fig = go.FigureWidget()




    context = {
        'y_fig': y_fig.to_html(),
        'plots': plots,
        'data_columns': data_columns,
        'categorical_data': categorical_data,
        'numerical_data': numerical_data,
    }

    return render(request, 'viz/visuals.html', context)


