import pandas as pd 
import numpy as np 
import math 
import statistics as st 
import random
import plotly.express as px 
import plotly.graph_objects as go
import datetime as dt 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

#to set env to a file go to file set work space as then select save 

banks=pd.read_csv('Hawaii_Banks_and_Credit_Unions.csv')
islandNames=banks['island'].unique()


#function to get the subset of points for the selected island
def islandChoice(island1):
    dfFun=banks[['X','Y','name','location']][banks['island']==island1]
    dfFun.index=list(range(0,len(dfFun)))
    dfFun.head()
    dfFun=dfFun.dropna()
    dfFun.index=list(range(0,len(dfFun)))
    return dfFun
#Dr. Day's function (from BZAN 544 spring 2021) function to get the best route for all points as starting points 
def route(dfExample):
    pointsList=list()
    for i in range(0,len(dfExample)):
        pointsList.append((dfExample['X'].iloc[i],dfExample['Y'].iloc[i]))
    allPaths = []
    allDists = []
    for i, startPtIndex in enumerate(pointsList):
        ptsLeft = pointsList.copy()
        aPath = []
        distance = 0
        aPath.append(ptsLeft.pop(i))
        while len(ptsLeft)>0:
            currPt = aPath[-1]
            distToCurrPt = [math.dist(currPt, pt) for pt in ptsLeft]
            minDistIndex = distToCurrPt.index(min(distToCurrPt))
            distance += distToCurrPt[minDistIndex]
            aPath.append(ptsLeft.pop(minDistIndex))
        allDists.append(distance)
        allPaths.append(aPath)
    bestDist = min(allDists)
    return allPaths[allDists.index(bestDist)]



app = dash.Dash('Bank App')

app.layout = html.Div(children=[
    html.H1('Prototype from: Jonathan Frederick'),
    
    html.H3('Banksy Aire Hawaii Islands route application'),

    html.P(children=[
    html.Label('Choose Island (scroll mouse forward or backward to zoom in or out)')
   ,dcc.Dropdown(id="input", options=[{'label': c, 'value': c } for c in islandNames] , value='Hawaii', multi=False)],	style = {'width': '300px'} ),

    dcc.Graph(id='output-graph1')
])

@app.callback(
    Output(component_id='output-graph1', component_property='figure'),
    [Input(component_id='input', component_property='value'),]
)


def updateMap(input_data):
    dfIsland=islandChoice(input_data)
    dfIs=pd.DataFrame(route(dfIsland),columns=['X','Y'])

    #not right merge the two together on a lat/long combo to get right seq of the names and locations 
    dfIs=pd.concat([dfIs,dfIsland[['name','location']]],axis=1)

    ints1=list(range(0,len(dfIs)))
    dfIs['Position']=list(map(str, ints1))

    latitude_list = list(dfIs['Y'])
    longitude_list = list(dfIs['X'])
    maenlong=st.mean(dfIs['X'])
    maenlat=st.mean(dfIs['Y'])
    
    fig = go.Figure(go.Scattermapbox(
        mode = "markers+lines",
        lon = longitude_list,
        lat = latitude_list,
        marker = dict(size= 10,color='blue'),
        line={'color': 'red'},
        textposition='top right',
        textfont=dict(size=16, color='black'),
        text=[dfIs['name'][i] + '<br>' + dfIs['Position'][i] for i in range(dfIs.shape[0])]))

    #figure out how to change the color of the lines while keeping the points blue , make the points a different color than the line 
    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': maenlong, 'lat': maenlat},
            'style': "stamen-terrain",
            'center': {'lon': maenlong, 'lat': maenlat},
            'zoom': 8})
    

    # this does not work: fig.update_layout(hovermode=dfIs.index)

    return fig 






if __name__ == '__main__':
    app.run_server(debug=True)
