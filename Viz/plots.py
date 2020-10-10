import plotly.express as px


def plot_bar(df, x_axis = None, y_axis=None, color = None, title:str=None) -> str:

    global fig
    try:
        fig = px.bar(df, x_axis, y_axis, color, title)
    except Exception as e:
        print('Insert Data set !')
    return fig


def plot_scatter(df, x_axis=None, y_axis=None):
    fig = px.scatter(df, x_axis, y_axis)
    return fig


def plot_line(df, x_axis=None, y_axis=None):
    fig = px.line(df, x_axis, y_axis)
    return fig


def plot_pie(df, values=None, names=None):
    fig = px.pie(df, values, names)
    return fig


def plot_box(df, x_axis=None, y_axis=None, color=None):
    fig = px.box(df, x_axis, y_axis, color, points='all', notched=True)
    return fig


def plot_box_algo(df, x_axis=None, y_axis=None):
    fig = px.box(df, x_axis, y_axis)
    fig.update_traces(quartilemethod="exclusive")
    return fig


def plot_histogram(df, x_axis=None):
    fig = px.histogram(df, x_axis, nbins=10)
    return fig


def density_heat_map(df, x_axis=None, y_axis=None) -> str:
    """

    :rtype: string
    """
    fig = px.density_heatmap(df, x_axis, y_axis, marginal_x='histogram', marginal_y="histogram")
    return fig