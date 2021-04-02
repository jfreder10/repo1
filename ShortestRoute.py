import pandas as pd 
import numpy as np 
import math 
import statistics as st 
import random
import plotly.express as px 
import plotly.graph_objects as go

def createPoints(num=5):
    this_data=[[random.uniform(-10,10),random.uniform(-10,10)] for i in range(num)]
    return pd.DataFrame(data=this_data,columns=['x','y'])

a=createPoints(3)

#step one initilize the starting path node 
path=[0]
dists=list(range(0,len(a)))
for i in range(0,len(a)):
    if(i in path):
        dists[i]=0
    else:
        dists[i]=sum(((a.iloc[path[0]]-a.iloc[i])**2)**.5)
dists
np.array(np.where(np.array(dists)!=0)).tolist()[0]
min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]])
#gets the next point that we should go to given we have initilized at point  k 
path.append(np.array(np.where(np.array(dists)==min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]]))).tolist()[0][0])
dists[np.array(np.where(np.array(dists)==min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]]))).tolist()[0][0]]


#step one initilize the starting path node 
path=[0]
dists=list(range(0,len(a)))
for i in range(0,len(a)):
    if(i in path):
        dists[i]=0
    else:
        dists[i]=sum(((a.iloc[path[0]]-a.iloc[i])**2)**.5)
dists
path.append(np.array(np.where(np.array(dists)==min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]]))).tolist()[0][0])
path 
#step 2 next iteration 
for i in range(0,len(a)):
    if(i in path):
        dists[i]=0
    else:
        dists[i]=sum(((a.iloc[path[0]]-a.iloc[i])**2)**.5)
dists
path.append(np.array(np.where(np.array(dists)==min(np.array(dists)[np.array(np.where(np.array(dists)!=0)).tolist()[0]]))).tolist()[0][0])
path
len(path)
len(a)

#while loop proto for function 
path=[0]
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

path
dists #store the sum of dists in a lists 
sumdists

#now make into function were starting index and data will be the argument 
#Function 
del path 
del dists 
del sumdists
del a
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

#test function 
a1=createPoints(30)
a1
df1=pd.DataFrame(bestithpath(a1,0)[0])
dfi=pd.DataFrame(bestithpath(a1,1)[0])
#cbind the next starting points best route to df1
df1=pd.concat([df1,pd.DataFrame(bestithpath(a1,1)[0])],axis=1)
#make an empty df with 1 col and i possible routes as nrows 
index=range(0,len(a1))
index
columns=['Distance']
distdf=pd.DataFrame(index=index, columns=columns)
distdf.iloc[0]=bestithpath(a1,0)[1]
distdf

a1=createPoints(30)
df1=pd.DataFrame(bestithpath(a1,0)[0])
index=range(0,len(a1))
columns=['Distance']
distdf=pd.DataFrame(index=index, columns=columns)
distdf.iloc[0]=bestithpath(a1,0)[1]
for i in range(1,len(a1)):
    dfi=pd.DataFrame(bestithpath(a1,i)[0])
    distdf.iloc[i]=bestithpath(a1,i)[1]
    df1=pd.concat([df1,dfi],axis=1)
df1
distdf
min(distdf.iloc[0:len(distdf),0])
np.array(np.where(distdf.iloc[0:len(distdf),0]==min(distdf.iloc[0:len(distdf),0]))).tolist()[0][0]
df1.iloc[0:len(df1),np.array(np.where(distdf.iloc[0:len(distdf),0]==min(distdf.iloc[0:len(distdf),0]))).tolist()[0][0]]

#now loop through all possible starting points and find best one and make this into function 
for i in range(0,len(a1)):
    print(bestithpath(a1,i))
    
del a1
#i think working but store as a df or something then get the min distance 
routes=list(range(0,len(a1))) #make into a df with len(a1) cols then make another df with only one col that will be the distances for each route
#can get the best route by finding which row in the distance df is the min and use this to get the jth col in the route df that is the best route 

del a1
del df1
del index 
del columns 
del distdf
del dfi
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

a=createPoints(15)
df=bestRoute(a,result='route')
df=pd.DataFrame(df)
#get data and make this into a visualization somehow, a plot that links the nodes together, scattter plot where the y var is the node and the x var is the order of 
#travel to the node (this will probably look stupid), output a datatable where the first row is the start the second row is the location to go to after 
#start and the 3rd row is the location to go to after the 2nd location, etc. 
#make input that restricts the number of places a driver can travel then only showw the top k max places 

#make some kind of visualization to use in a dash app, see if px has a table vis
#cant use plotly exp but go has way to use df from a pandas df 

#good proto for the best app 
df.columns=['Route']
df.Route
fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Route],
               fill_color='lavender',
               align='left'))
])
fig.show()


df.iloc[0:len(df),0]


