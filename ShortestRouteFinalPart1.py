import pandas as pd 
import numpy as np 
import math 
import statistics as st 
import random
import plotly.express as px 

#function that creates random points 
def createPoints(num=5):
    this_data=[[random.uniform(-10,10),random.uniform(-10,10)] for i in range(num)]
    return pd.DataFrame(data=this_data,columns=['x','y'])

#function that gets the best path per any starting point 
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

#function that gives the best path given all possible starting points 
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