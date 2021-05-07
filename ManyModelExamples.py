import sklearn
import pandas as pd 
import numpy as np
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statistics as st
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from statsmodels  import regression as reg
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#import a dataset for classification and one for regression, fit a model to both data sets to give tempelate code for future include data with cat vars as x vars

#churn data from 545 is class data predict churn 
churn=pd.read_csv(r'C:\Users\fredr\Documents\assignment2\FinalData.csv')
churn.columns
ChurnBuild=churn[churn.columns[[i for i,n in  enumerate(churn.dtypes) if n != 'object']]]
ChurnBuild['churn']=churn['churn']
churn.drop(['churn'],axis=1,inplace=True)
pd.get_dummies(churn[churn.columns[[i for i,n in  enumerate(churn.dtypes) if n == 'object']]], dummy_na=True)
ChurnBuild=pd.concat([ChurnBuild,pd.get_dummies(churn[churn.columns[[i for i,n in  enumerate(churn.dtypes) if n == 'object']]], dummy_na=True)],axis=1)
ChurnBuild.head()
ChurnBuild.fillna(ChurnBuild.mean(),inplace=True)
ChurnBuild.head()


#cab data is reg data predict price 
cabs=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\cabs.csv')
cabs.info
cabs.columns
#remove the id cols 
cabs=cabs.drop(['id','product_id'],axis=1)
#make cat cols into dummy vars 
[i for i, n in enumerate(cabs.dtypes) if n != 'object']
cabsBuild=cabs[cabs.columns[[i for i, n in enumerate(cabs.dtypes) if n != 'object']]]
#append the numeric cols and the dummy cols together 
cabsBuild=pd.concat([cabsBuild,pd.get_dummies(cabs[cabs.columns[[i for i, n in enumerate(cabs.dtypes) if n == 'object']]])],axis=1)
cabsBuild.fillna(cabsBuild.mean(),inplace=True)

#define function that will create equal to expand grid in R 
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}





##
#boosting
##

#boosting classification 
train, test =train_test_split(ChurnBuild,test_size=.2)
y_train=train['churn']
x_train=train.drop(['churn'], axis=1)
x_train.columns

#set the parameters 
n_estimators=np.arange(300, 350, 50) #the number of trees to fit 
max_depth=np.arange(3, 4, 1)
min_samples_split=np.arange(3,4,1)
learning_rate=np.arange(0.001,0.002,0.001)
a=expandgrid(n_estimators,max_depth, min_samples_split,learning_rate)
params=pd.DataFrame.from_dict(a)
len(params)
estValAcc=list(range(0,len(params)))

for i in range(0,len(params)):
    scores = cross_val_score(HistGradientBoostingClassifier(min_samples_leaf=params['Var3'].iloc[i],
    max_depth=params['Var2'].iloc[i],
    learning_rate=params['Var4'].iloc[i],max_iter=params['Var1'].iloc[i]).fit(x_train, y_train), 
    x_train, y_train, cv=4,scoring='accuracy')
    acc=st.mean(scores)
    estValAcc[i]=acc

estValAcc


#boosting for regression 
train, test =train_test_split(cabsBuild,test_size=.5)
y_train=train['price']
x_train=train.drop(['price'], axis=1)
x_train.columns

estValRmse=list(range(0,len(params)))
for i in range(0,len(params)):
    scores = cross_val_score(HistGradientBoostingRegressor(min_samples_leaf=params['Var3'].iloc[i],
    max_depth=params['Var2'].iloc[i],
    learning_rate=params['Var4'].iloc[i],max_iter=params['Var1'].iloc[i]).fit(x_train, y_train), 
    x_train, y_train, cv=4,scoring='neg_mean_squared_error')
    rmse=st.mean((scores*-1) ** (1/2))
    estValRmse[i]=rmse

estValRmse



##
#Random Forest 
##

#random forest for classification 
train, test =train_test_split(ChurnBuild,test_size=.2)
y_train=train['churn']
x_train=train.drop(['churn'], axis=1)
x_train.columns

#set the possible number of trees and the number of random vars considered for a split in each tree 
n_trees=np.arange(300, 350, 500) 
mtry=np.arange(5, 6, 1) 

#set the grid that has all possible combinations of ntress and mtry and make list to loop into to store est
#validation errors for each parameter setting of the random forest 
a=expandgrid(n_trees,mtry)
params=pd.DataFrame.from_dict(a)
params
estValAcc=list(range(0,len(params)))

#n_jobs sets the amount of cores in computer to do paralell processing (left out for this but add it after max_featurs arg )
for i in range(0,len(params)):
    scores=cross_val_score(RandomForestClassifier(max_depth=1, random_state=0, n_estimators=params.iloc[i,0],
    max_features=params.iloc[i,1]).fit(x_train, y_train), x_train,y_train,cv=4,scoring='accuracy')
    rmse=st.mean((scores*-1) ** (1/2))
    estValAcc[i]=rmse

estValAcc

#random forest for regression 
train, test =train_test_split(cabsBuild,test_size=.5)
y_train=train['price']
x_train=train.drop(['price'], axis=1)
x_train.columns

estValRmse=list(range(0,len(params)))
for i in range(0,len(params)):
    scores=cross_val_score(RandomForestRegressor(max_depth=1, random_state=0, n_estimators=params.iloc[i,0],
    max_features=params.iloc[i,1]).fit(x_train, y_train), x_train,y_train,cv=4,scoring='neg_mean_squared_error')
    rmse=st.mean((scores*-1) ** (1/2))
    estValRmse[i]=rmse

estValRmse



##
#svm
##

#svm for class
train, test =train_test_split(ChurnBuild,test_size=.2)
y_train=train['churn']
x_train=train.drop(['churn'], axis=1)
x_train.columns

Cost=np.arange(.10,.5,.10)
estValAcc=list(range(0,len(Cost)))
estValAcc
for i in range(0,len(C)):
    clf=svm.SVC(kernel='linear', C=Cost[i])
    score=cross_val_score(clf.fit(x_train, y_train),x_train,y_train,cv=4,scoring='accuracy')
    estValAcc[i]=np.mean(score)
estValAcc



#svm for reg 
train, test =train_test_split(cabsBuild,test_size=.5)
y_train=train['price']
x_train=train.drop(['price'], axis=1)
x_train.columns

Cost=np.arange(.10,.5,.10)
estValRmse=list(range(0,len(C)))
estValRmse
for i in range(0,len(C)):
    clf=svm.SVR(kernel='linear', C=Cost[i])
    score=cross_val_score(clf.fit(x_train, y_train),x_train,y_train,cv=4,scoring='neg_mean_squared_error')
    rmse=st.mean((scores*-1) ** (1/2))
    estValRmse[i]=rmse
    
estValRmse




##
#nn
##



#nn for class 

train, test =train_test_split(ChurnBuild,test_size=.2)
y_train=train['churn']
x_train=train.drop(['churn'], axis=1)
x_train.columns

alpha=np.arange(.0005,.0006,.0001)
randState=np.arange(1,3,1)
hidden1=np.arange(1,4,1)
hidden2=np.arange(2,4,1)
a=expandgrid(alpha,randState,hidden1,hidden2)
params=pd.DataFrame.from_dict(a)
params


estValAcc=list(range(0,len(params)))
for i in range(0,len(params)):
    clf = MLPClassifier(solver='lbfgs', alpha=params['Var1'].iloc[i],hidden_layer_sizes=(params['Var3'].iloc[i], params['Var4'].iloc[i]), 
    random_state=params['Var2'].iloc[i])
    score=cross_val_score(clf.fit(x_train, y_train),x_train,y_train,cv=4,scoring='accuracy')
    estValAcc[i]=np.mean(score)
estValAcc

#nn for reg 
train, test =train_test_split(cabsBuild,test_size=.5)
y_train=train['price']
x_train=train.drop(['price'], axis=1)
x_train.columns

estValRmse=list(range(0,len(params)))
for i in range(0,len(params)):
    clf = MLPRegressor(solver='lbfgs', alpha=params['Var1'].iloc[i],hidden_layer_sizes=(params['Var3'].iloc[i], params['Var4'].iloc[i]), 
    random_state=params['Var2'].iloc[i])
    scores=cross_val_score(clf.fit(x_train, y_train),x_train,y_train,cv=4,scoring='neg_mean_squared_error')
    rmse=st.mean((scores*-1) ** (1/2))
    estValRmse[i]=rmse
estValRmse


##
#knn
##


#knn for class 



#knn for reg 




##
#lasso
##


#lasso for logistic reg (class)


#lasso for reg 



##
#kmeans for unsupervised learning 
##
kmeans=KMeans(n_clusters=2, random_state=0).fit(cabsBuild)
kmeans.labels_

ChurnKmeans=pd.get_dummies(ChurnBuild)
kmeans=KMeans(n_clusters=2, random_state=0).fit(ChurnKmeans)
kmeans.labels_


##
#DBSCAN for unsupervised learning 
##

#use np.array so it runs faster and set n_jobs to something 
np.array(ChurnKmeans)
clustering = DBSCAN(eps=3, min_samples=2,n_jobs=6).fit(np.array(ChurnKmeans))
clustering.labels_



##
#principal components 
##
xDF = StandardScaler().fit_transform(ChurnKmeans)
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(xDF)
dir(principalComponents)
pcaDF=pd.DataFrame(principalComponents)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))







