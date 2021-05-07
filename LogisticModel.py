import pandas as pd
import numpy as np
import statsmodels as stm #loads statsmodels for linear regresssion 
import matplotlib.pyplot as plt
from statsmodels  import regression as reg
import statsmodels.api as sm
import seaborn as sns
import itertools
import scipy.stats as spst #package for probability modeling 
import statistics as st
import sklearn as skl #imports sklearn 
import statsmodels.api as regMods
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt 
import plotly.express as px 
from math import comb

heart=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\heart.csv')
heart['intercept']=1
heart.info() #target is neumeric 
x=np.array(heart['thalach']).reshape(-1,1)
x
y=np.array(heart['target'])
y
#with stats mods

x=sm.add_constant(x) #adds intercept to x matrix 
x[0:len(x),1]
model = sm.Logit(y, x)
result = model.fit(method='newton')
result.params
result.predict(x)
min(result.predict(x))
result.summary()
dir(result)
result.llf

#plot the output 0 or one on y axis an x as x axis then plot the predicted probs as a line through the plot 
#plots the logistic reg preds as y to x 
fig=px.scatter(x=x[0:len(x),1],y=result.predict(x))
fig.show()
np.mean(heart['age'][heart['target']==0])
np.mean(heart['age'][heart['target']==1]) 
#fit the log odds to x and perform model diagonstics 
logodds=np.log(result.predict(x)/(1-result.predict(x)))
logodds
fig=px.scatter(x=x[0:len(x),1],y=logodds)
fig.show()
#get the satuarated model for each unique value of x 
#get the prop of the values in target that ==1 for each unique value of x 
heart['thalach'].unique()
len(heart[heart['thalach']==heart['thalach'].unique()[0]][heart['target'][heart['thalach']==heart['thalach'].unique()[0]]==1])/len(heart[heart['thalach']==heart['thalach'].unique()[0]])
saturated=np.zeros(len(heart['thalach'].unique()))
saturated[0]
for i in range(0,len(heart['thalach'].unique())):
    saturated[i]=len(heart[heart['thalach']==heart['thalach'].unique()[i]][heart['target'][heart['thalach']==heart['thalach'].unique()[i]]==1])/len(heart[heart['thalach']==heart['thalach'].unique()[i]])

saturated
#plot the saturate probs given each value of x to the fitted probs 
#fit mod on the unique values of x to compare to saturated model 
xunique=np.array(heart['thalach'].unique())
x=sm.add_constant(xunique)
len(result.predict(x))
len(saturated)
df1=pd.DataFrame({'x':heart['thalach'].unique(),'fitted':result.predict(x),'model':'log'})
df2=pd.DataFrame({'x':heart['thalach'].unique(),'fitted':saturated,'model':'saturated'})
dfBoth=pd.concat([df1,df2])
#not a good fit the models fitted values do not match up with the saturated probs given values of x 
fig=px.scatter(x=dfBoth['x'],y=dfBoth['fitted'],color=dfBoth['model'])
fig.show()


#EX if 50 possible x vars but only want 10 in model then choose(50,10) arn number of possible unique variable combos possible, then if want 
# combos of all 3 two way interactions among all possible predictors of 10 get the ith predictor subset and do choose(10,2) these would be the number of possible 
#2 way interactions, then of this list select 3 at a time to fit in model with other 10 predictors of ith combo subset, 
#so total number of vars would be: choose(50,10)*choose(len(choose(10,2)),3)
#  


#figure out how to select all possible 2 way interactions among variables given a vector of possible x vars
#choose len(x), 2 1st possible 2 way interactions  
#(len(x)-1, 2)-1 2nd possible interactions
heart.columns
xvars=np.array(['age','sex','cp','chol'],dtype=str)
comb(4,2) #6 possible combos of a 2 way interaction for the first possible interaction 
itertools.combinations(xvars, 2)
#gets all possible 2 way interactions for the first interaction variable 
twoways=[i for i in itertools.combinations(xvars, 2)]
#fit a model with all x vars predicting target and each possible pair of the 2 way interactions (2 2way interactions) choose 6,2 
#gets all possible combos of 2 two way interaction vars for a model 
twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)]
len(twoTwoWays)
heart[twoTwoWays[0][0]] #gets the columns for the first interaction in the first combo of all possible 2 interactions 

#loop through combos of all possible x vars with size equal to k as outer loop 
xvars=np.array(['age','sex','cp','chol','oldpeak','thalach'],dtype=str)
predictors=[i for i in itertools.combinations(xvars, 4)]
len(predictors)
preds_i=np.array(predictors[0])
preds_i=np.append(preds_i,'intercept')
x=np.array(heart[np.array(preds_i)])
y=np.array(heart['target'])

model = sm.Logit(y, x)
result = model.fit(method='newton') #dtore this as the ith element in kmods list 
result.params
result.predict(x)

kmods=list()
for i in range(0,len(predictors)):
    preds_i=np.array(predictors[i])
    preds_i=np.append(preds_i,'intercept')
    x=np.array(heart[np.array(preds_i)])
    model = sm.Logit(y, x)
    result = model.fit(method='newton')
    kmods.append(result)

len(kmods)
kmods[0].summary()


#for each ith possible var combo loop through the j interactions the size k out of the number of selected vars for the ith combo 
#then need to loop through the interactions selected to create the interaction vars 
predictors=[i for i in itertools.combinations(xvars, 4)]
predictors[0]
#in outter loop right before inner 
twoways=[i for i in itertools.combinations(predictors[0], 2)]
len(twoways) 
twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)]
twoTwoWays
twoTwoWays[0] #make 2 new vars that are age*sex and age*cp 
interactions_j=np.array(twoTwoWays[0])
interactions_j

heart[interactions_j[0]] #get product of these 2 cols as the names pasted together 
 #this initilizes the interaction column
intcols=list()
for m in range(0,len(interactions_j)):
    heart[interactions_j[m][0]+'_'+interactions_j[m][1]]=heart[interactions_j[0]].iloc[0:len(heart[interactions_j[m]]),0]*heart[interactions_j[m]].iloc[0:len(heart[interactions_j[m]]),1]
    intcols.append(pd.DataFrame(heart[interactions_j[m][0]+'_'+interactions_j[m][1]]).columns)
heart
np.array(intcols)

preds_i=np.array(predictors[0])
predictors
preds_i=np.append(preds_i,'intercept')
np.append(preds_i,np.array(intcols)) #put 
x=np.array(heart[np.array(preds_i)])
ith_interaction_mod=list()




for j in range(0,len(twoTwoWays)):
    interactions_j=np.array(twoTwoWays[j])
    intcols=list()
    for m in range(0,len(interactions_j)):
        heart[interactions_j[m][0]+'_'+interactions_j[m][1]]=heart[interactions_j[0]].iloc[0:len(heart[interactions_j[m]]),0]*heart[interactions_j[m]].iloc[0:len(heart[interactions_j[m]]),1]
        intcols.append(pd.DataFrame(heart[interactions_j[m][0]+'_'+interactions_j[m][1]]).columns)
        
    np.append(preds_i,np.array(intcols))
    x=np.array(heart[np.array(preds_i)])
    model = sm.Logit(y, x)
    heart=heart.drop([interactions_j[m][0]+'_'+interactions_j[m][1]],axis=1) 
    result = model.fit(method='newton')
    ith_interaction_mod.append(result)
heart
len(ith_interaction_mod) #15 because we only went through one possible combo 

#wrap code in outer loop for selecting the ith subset of predictors size = k 
#gets all models for a subset of xvars = k and for all possible 2 two way interactions 
predictors=[i for i in itertools.combinations(xvars, 4)] #loop through as outer loop 
twoways=[i for i in itertools.combinations(predictors[0], 2)] #set this to get the number of m 2 way interaction combos 
twoTwoWays=[i for i in itertools.combinations(np.array(twoways),2)] #loop through in inner with the predictors set to i for all j 
#innner loop of the inner loop creates vars so can append to current x vars from outer loop 
allmods=list()
for i in range(0,len(predictors)):
    preds_i=np.array(predictors[i])
    for j in range(0,len(twoTwoWays)):
        interactions_j=np.array(twoTwoWays[j])
        intcols=list()
        for m in range(0,len(interactions_j)):
            heart[interactions_j[m][0]+'_'+interactions_j[m][1]]=heart[interactions_j[0]].iloc[0:len(heart[interactions_j[m]]),0]*heart[interactions_j[m]].iloc[0:len(heart[interactions_j[m]]),1]
            intcols.append(pd.DataFrame(heart[interactions_j[m][0]+'_'+interactions_j[m][1]]).columns)
            
        np.append(preds_i,np.array(intcols))
        x=np.array(heart[np.array(preds_i)])
        model = sm.Logit(y, x)
        heart=heart.drop([interactions_j[m][0]+'_'+interactions_j[m][1]],axis=1) 
        result = model.fit(method='newton')
        allmods.append(result)
len(allmods) #225 possible models 
comb(6,4)
comb(6,2)
#comb(6,4)*comb(6,2)=225 


#make into a function where the inputs are: x (array from np), y (one var array from np), arg to choose if linear or logistic reg 
















len(twoTwoWays)
heart[twoTwoWays[0][0]]







np.exp(5)/(1+np.exp(5)) #=.993 is prob 
#odds of prob
0.993/(1-0.993)
#log odds of prob 
np.log(0.993/(1-0.993)) # equals 5 (just about) abd 5 was the sum of b0+b1x1 so log(odds)=bo+b1x1 
np.log(np.exp(5)/np.exp(1+5))




