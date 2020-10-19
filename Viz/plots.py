import dash
from django_plotly_dash import DjangoDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc



def graph(app, df):
    graph_card = dbc.Card([
            dbc.CardBody([
                html.H4("Data Visualization"),
                html.Hr(),
                dcc.Graph(id="unique_graph")
            ],
            )
        ],
        style={"height":700},
        color="white",
        inverse=False,
        outline=False,
    )

    option_card = dbc.Card([
            dbc.CardBody([
                html.H4("Select types "),
                html.Br(),
                dcc.Dropdown(id="graph_type",
                    options=[
                        {'label': 'Scatter', 'value': 'SCATTER'},
                        {'label': 'Bar', 'value': 'BAR'},
                        {'label': 'Line', 'value': 'LINE'},
                        {'label': 'Area', 'value': 'AREA'},
                        {'label': 'HeatMap', 'value': 'HEET'},
                        {'label': 'Pie', 'value': 'PIE'},
                        {'label': 'Box', 'value': 'BOX'},
                        {'label': 'Histogram', 'value': 'HIST'},
                        {'label': 'Scatter Matrix', 'value': 'SMX'},
                        {'label': 'Violin', 'value': 'VIOLIN'},
                    ],
                    value='BAR',
                    searchable=True,
                    clearable=False,
                    className="form-dropdown",
                    persistence='string',
                    persistence_type='memory',
                ),
                html.Br(),
                dcc.Dropdown(id="x_axis",
                    options=[{'label':x, 'value':x} for x in df.columns],
                    placeholder="X axis",
                    multi=False,
                    searchable=True,
                    className="form-dropdown",
                    persistence='string',
                    persistence_type='local',
                ),
                html.Br(),
                dcc.Dropdown(id="y_axis",
                    options=[{'label':x, 'value':x} for x in df.columns],
                    placeholder="Y axis", searchable=True, clearable=False,
                ),
                html.Br(),
            ])
        ],
        style={"height":700},
        color="white",
    )
    app.layout = html.Div([
        dbc.Row([
            dbc.Col(graph_card, width=9),
            dbc.Col(option_card, width=3)
        ],
        style={"margin-top": 15},
        justify="around"),
    ])
    @app.callback(
        Output('unique_graph','figure'),
        [Input('graph_type','value'),
        Input('x_axis','value'),
        Input('y_axis','value')]
    )
    def build_graph(graph_type, x_axis, y_axis):
        dff=df

        if graph_type == 'LINE':
            fig = px.line(dff, x=x_axis, y=y_axis, color=None, height=600)
            fig.update_layout(yaxis={'title':y_axis},
                        title={'text':x_axis+' vs '+ y_axis,
                        'font':{'size':28},'x':0.5,'xanchor':'center'})
            
        elif graph_type == 'SCATTER':
            fig = px.scatter(dff, x=x_axis, y=y_axis, color=None, height=600)
            fig.update_layout(yaxis={'title':y_axis},
                            title={'text':x_axis+' vs '+ y_axis,
                            'font':{'size':28},'x':0.5,'xanchor':'center'})

        elif graph_type == 'BAR':
            fig = px.bar(dff, x=x_axis, y=y_axis, color=None, height=600)
            fig.update_xaxes(type='category')
            fig.update_layout(yaxis={'title':y_axis},
                            title={'text':x_axis+' vs '+ y_axis,
                            'font':{'size':28},'x':0.5,'xanchor':'center'})
        
        elif graph_type == 'AREA':
            fig = px.area(dff, x=x_axis, y=y_axis, color=None, height=600)
            fig.update_layout(yaxis={'title':y_axis},
                            title={'text':x_axis+' vs '+ y_axis,
                            'font':{'size':28},'x':0.5,'xanchor':'center'})

        elif graph_type == 'HEET':
            fig = px.density_heatmap(dff, x=x_axis, y=y_axis,  nbinsx=20, nbinsy=20, marginal_x="histogram", marginal="histogram")

        elif graph_type == 'BOX':
            fig = px.box(dff, x=x_axis, y=y_axis, color=None, height=600)
            fig.update_layout(yaxis={'title':y_axis},
                            title={'text':x_axis+' vs '+ y_axis,
                            'font':{'size':28},'x':0.5,'xanchor':'center'})
        
        elif graph_type == 'PIE':
            fig = px.pie(dff, values=x_axis, names=y_axis, height=600)
            fig.update_layout(yaxis={'title':y_axis},
                            title={'text':x_axis+' vs '+ y_axis,
                            'font':{'size':28},'x':0.5,'xanchor':'center'})

        elif graph_type == 'HIST': 
            fig = px.histogram(dff, x=x_axis, y=y_axis,marginal="box", color=None, height=600)  # can be box or violin
            fig.update_layout(yaxis={'title':y_axis},
                            title={'text':x_axis+' vs '+ y_axis,
                            'font':{'size':28},'x':0.5,'xanchor':'center'})

        elif graph_type == 'SMX':
            fig = px.scatter_matrix(dff, color=None)
            fig.update_layout(yaxis={'title':y_axis},
                                title={'text':x_axis+' vs '+ y_axis,
                                'font':{'size':28},'x':0.5,'xanchor':'center'})
        
        elif graph_type == 'VIOLIN': 
            fig = px.violin(dff, x=x_axis, y=y_axis, box=True, color=None, height=600)  # can be box or violin
            fig.update_layout(yaxis={'title':y_axis},
                            title={'text':x_axis+' vs '+ y_axis,
                            'font':{'size':28},'x':0.5,'xanchor':'center'})

        return fig