#math and df modules 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import itertools
from math import exp
from distfit import distfit
#sklearn modules 
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#visualizations modules 
import plotly.express as px
import plotly.figure_factory as ff 
#pickle to store models in so dont need to rerun 
import pickle 



##
##data explore
##
#predict sales price from the vars 
test=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\RealEstateKaggle\test.csv')

#import in as Rest and make var that is train do smae for test and append ten store var in array then drop and rerun and split before build models 
rEst=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\RealEstateKaggle\train.csv')
#set y from train 
y=np.array(rEst['SalePrice'])
rEst.drop(['SalePrice'],axis=1, inplace=True)
rEst.drop(['Id'],inplace=True,axis=1)
rEst['Data']='Train'

testIds=np.array(test['Id'])
test.drop(['Id'],inplace=True,axis=1)
test['Data']='Test'
rEst=pd.concat([rEst,test],axis=0) 
testTrainIndex=rEst['Data']
rEst.drop(['Data'],inplace=True,axis=1)


y=np.array(rEst['SalePrice'])
rEst.drop(['SalePrice'],axis=1, inplace=True)

#rEst=pd.read_csv('RealEstClean.csv')
rEst.columns
#rEst.drop(['Unnamed: 0'],inplace=True,axis=1)
rEstInitial.drop(['Id'],inplace=True,axis=1)
rEstInitial.head()
len(rEstInitial)
rEstInitial.columns[0:90]
rEstInitial.info()

##################################################################################
##
##create new variabes 
##

##################################################################################
##
#kmeans 
##
#fit one or more kmeans schemes to the data use sklearn package
#loop through possible k and store the wcss in a np.array
rEstNum=pd.get_dummies(rEst, dummy_na=True)
rEstNum.fillna(rEstNum.mean(),inplace=True)
possibleK=np.arange(2,8,1)
possibleK
wcss=np.array(np.zeros(len(possibleK)),dtype='float')

kmeans=KMeans(n_clusters=2, random_state=0).fit(rEstNum)
dir(kmeans)
#this is the WCSS 
kmeans.inertia_

for i in range(0,len(possibleK)):
    kmeans=KMeans(n_clusters=possibleK[i], random_state=0).fit(rEstNum)
    wcss[i]=kmeans.inertia_

wcss

#use plotly express to plot wcss as y and possible k as x 
fig=px.scatter(x=possibleK,y=wcss)
fig.show()
#use k=4 
kmeans=KMeans(n_clusters=4, random_state=0).fit(rEstNum)
rEstNum.columns 
rEst['kmeans4']=kmeans.labels_

#loop through several clustering methods and make many vars based on best scheme per each one 





##################################################################################
##
#pdf values 
##
#fit each possible distro to each col then get likelihood then select the distro with the highest likelihood 
#check distros with dist fit, this is the best way 
#loop through all the vars and select the distro with the lowest rss then store these in a list, use getattr to input data to them with stats 
#use P as the new feature variable 
rEstPdfs=rEst[rEst.columns[[[i for i,n in enumerate(rEst.dtypes) if n !='object']]]]
rEstPdfs.fillna(rEstPdfs.mean(),inplace=True)

dist = distfit(todf=True)
cols=rEstPdfs.columns

#gets the pdf values for each var from the distro the ith var is closest to 
for i in range(0,len(cols)):
    distros_i=dist.fit_transform(rEstPdfs[cols[i]])
    out=dist.predict(np.array(rEstPdfs[cols[i]]))
    rEst['pdf'+cols[i]]=out['df']['P']

len(rEst.columns) #117 vars now 

#do same but for the distribution the varible is farthest from to see if this creates a lot of var in data among values of y 
##################################################################################
##
#save data here:
##
rEst.to_csv('C:/Users/fredr/Documents/StatTool/PythonCode/FunctionsAndCoolStuff/RealEstateKaggle/RealEstClean.csv')

rEst.head()


##################################################################################
##
#set train and test 
##
rEstBuild['kmeans4']=rEstBuild['kmeans4'].astype(str)
rEstBuild=pd.get_dummies(rEst)
rEstBuild.fillna(rEstBuild.mean(),inplace=True)

train=rEstBuild.iloc[np.where(testTrainIndex=='Train')]
test=rEstBuild.iloc[np.where(testTrainIndex=='Test')]

y_train=y
x_train=train 
x_test=test 
len(x_train.columns)
len(x_test.columns)

x_train.columns
y_train=np.log(y_train)
#make kmeans into cat var 



##
#fit and evaluate models
##

def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

##################################################################################
##
#neural network
##
#make a function that will take on the number of hidden layer and the range of values for each node size for each layer and out put will be a array of all possible 
# combos where the cols will represent the layers 

#parameters for nn with 3 hidden layers 
alpha=np.arange(.0002,.0003,.0001)
randState=np.arange(1,2,1)
hidden1=np.arange(5,6,1)
hidden2=np.arange(8,9,1)
hidden3=np.arange(4,5,1)
iters=np.arange(200,300,100)
a=expandgrid(alpha,randState,hidden1,hidden2,hidden3,iters)
params=pd.DataFrame.from_dict(a)
params.head()
len(params) #

#put n_jobs in clf.fit 
estValRmse=list(range(0,len(params)))
for i in range(0,len(params)):
    clf = MLPRegressor(solver='lbfgs', alpha=params['Var1'].iloc[i],hidden_layer_sizes=(params['Var3'].iloc[i], params['Var4'].iloc[i],params['Var5'].iloc[i]), 
    random_state=params['Var2'].iloc[i], max_iter=params['Var6'].iloc[i])
    scores=cross_val_score(clf.fit(x_train, y_train),x_train,y_train,cv=4,scoring='neg_mean_squared_error',n_jobs=10)
    rmse=st.mean((scores*-1) ** (1/2))
    if( (i==0) or (min(estValRmse)>rmse) ):
        estValRmse[i]=rmse
        bestMod=clf
    else:
        estValRmse[i]=rmse
        continue



min(estValRmse)  #min is .23, tune: alpha next
bestMod.get_params
bestNN=bestMod
#saves the model in directory, have set workspace 
filename = 'finalizedNN_model.sav'
pickle.dump(bestNN, open(filename, 'wb'))
#del bestNN

#loads model from directory workspace is in 
loaded_bestNN = pickle.load(open(filename, 'rb'))
loaded_bestNN.get_params
result = loaded_model.score(X_test, Y_test)

##################################################################################
##
#rf
##
n_trees=np.arange(1000, 1500, 100) 
mtry=np.arange(25, 35, 1) 
a=expandgrid(n_trees,mtry)
params=pd.DataFrame.from_dict(a)
len(params)

estValRmse=list(range(0,len(params)))
for i in range(0,len(params)):
    clf=RandomForestRegressor(max_depth=1, random_state=0, n_estimators=params.iloc[i,0],
    max_features=params.iloc[i,1])
    scores=cross_val_score(clf.fit(x_train, y_train), x_train,y_train,cv=4,scoring='neg_mean_squared_error',n_jobs=10)
    rmse=st.mean((scores*-1) ** (1/2))
    if( (i==0) or (min(estValRmse)>rmse) ):
        estValRmse[i]=rmse
        bestMod=clf
    else:
        estValRmse[i]=rmse
        continue
#get var importance to see if any new vars are important 
estValRmse
min(estValRmse) #.28 to high 
bestMod.get_params
#make best param finalmod 

imp=pd.DataFrame(bestMod.feature_importances_)
imp['variable']=x_train.columns
imp.columns=['importance','variable']
imp=imp.sort_values('importance', ascending=False)
fig=px.bar(imp,x='variable',y='importance')
fig.show()

imp[imp['variable']=='kmeans4']

#unlog predictions with exp(predictions) exp is function from math package 

##################################################################################
##
#hist boosting
##
n_estimators=np.arange(500, 700, 50) #the number of trees to fit 
max_depth=np.arange(9, 12, 1)
min_samples_split=np.arange(3,4,1)
learning_rate=np.arange(0.0083,0.0087,0.0001)
a=expandgrid(n_estimators,max_depth, min_samples_split,learning_rate)
params=pd.DataFrame.from_dict(a)
params=np.array(params)
params
len(params)
estValRmse=list(range(0,len(params)))



for i in range(0,len(params)):

    clf=HistGradientBoostingRegressor(min_samples_leaf=int(params[i,2]),
    max_depth=int(params[i,1]),
    learning_rate=params[i,3],max_iter=int(params[i,0]), warm_start=True)

    scores = cross_val_score(clf.fit(np.array(x_train), np.array(y_train)), 
    x_train, y_train, cv=4,scoring='neg_mean_squared_error',n_jobs=14)

    rmse=st.mean((scores*-1) ** (1/2))

    if( (i==0) or (min(estValRmse)>rmse) ):
        estValRmse[i]=rmse
        bestMod=clf
    else:
        estValRmse[i]=rmse
        continue
    

min(estValRmse)
len(estValRmse)
estValRmse
len(params)
np.where(np.array(estValRmse)==min(estValRmse))
params.iloc[23]
bestMod.get_params()
bestBoost=bestMod


#saves the model in directory, have set workspace 
filename = 'bestBoost_model.sav'
pickle.dump(bestBoost, open(filename, 'wb'))
#del bestNN

#loads model from directory workspace is in 
loaded_bestBoost = pickle.load(open(filename, 'rb'))
loaded_bestBoost.get_params



#if len of df set to a df1 then append to a df2 to store all in 

#set bestmod.get_parameters to a df then set as another df, then set a var to the run number of fitting models (set this in loop, update ate end var+=1)
#var -1 will be the iteration you just did and var will be the iteration that will be the next run 
#make if else if var ==1 then initilize best paramDoc to best mod params else set another df to best params and then append into the initial one 



#for var selection find a good model then do k fold cv with each possible var choice of size k out of all n variables

