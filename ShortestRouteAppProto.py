import pandas as pd 
import numpy as np 
import math 
import statistics as st 
import random
import plotly.express as px 
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output


def createPoints(num=5):
    this_data=[[random.uniform(-10,10),random.uniform(-10,10)] for i in range(num)]
    return pd.DataFrame(data=this_data,columns=['x','y'])

def bestithpath(a,index1):
    path=[index1]
    dists=list(range(0,len(a)))
    sumdists=0
    while(len(path)<len(a)):
        for i in range(0,len(a)):
            if(i in path):
                dists[i]=0
            else:
                dists[i]=sum(((a.iloc[path[0]]-a.iloc[i])**2)**.5)
        path.append(np.array(np.where(np.array(dists)==min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]]))).tolist()[0][0])
        sumdists+=min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]])
    return path, sumdists


def bestRoute(a1,result='mindist'):
    df1=pd.DataFrame(bestithpath(a1,0)[0])
    index=range(0,len(a1))
    columns=['Distance']
    distdf=pd.DataFrame(index=index, columns=columns)
    distdf.iloc[0]=bestithpath(a1,0)[1]
    for i in range(1,len(a1)):
        dfi=pd.DataFrame(bestithpath(a1,i)[0])
        distdf.iloc[i]=bestithpath(a1,i)[1]
        df1=pd.concat([df1,dfi],axis=1)
    #get the min of distdf and the row of that min value so that can get the col for this in df1 as the best path 
    mindist=min(distdf.iloc[0:len(distdf),0])
    bestpath=df1.iloc[0:len(df1),np.array(np.where(distdf.iloc[0:len(distdf),0]==min(distdf.iloc[0:len(distdf),0]))).tolist()[0][0]]
    if(result=='mindist'):
        return mindist  
    else:
        return bestpath

numpoints=range(5,1000)

app = dash.Dash('Shortest Route App')

app.layout = html.Div(children=[
    html.H1('Schedule Route Problem'),
    html.Div(children='''
        Choose number of random locations:
    '''),

    dcc.Dropdown(id="input", options=[{'label': c, 'value': c } for c in numpoints] , value=5), 
    html.Div(children='''
        Distance then route:
    '''),  
    #html.Div below is the out put for the schedule for the best route 
    html.Div(id='output-graph1'),
    dcc.Graph(id='output-graph2'),
])

@app.callback(
    Output(component_id='output-graph1', component_property='children'),
    Output(component_id='output-graph2', component_property='figure'),
    [Input(component_id='input', component_property='value'),]
)
#make a function to indicate that the input and output are connected 
def update_route1(input_data):
    a=createPoints(input_data)
    #put the table here figure if need dcc.Graph in layout code or something else 
    df=bestRoute(a,result='route')
    df=pd.DataFrame(df)
    df.columns=['Route']
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
        cells=dict(values=[df.Route],
               fill_color='lavender',
               align='left'))
])
    return 'Distance: {}'.format(bestRoute(a,result='mindist')), fig 





if __name__ == '__main__':
    app.run_server(debug=True)