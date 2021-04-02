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

gdb=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\BZAN540\Homework\HW1\GoodBellyData.csv')
gdb=gdb[gdb.columns[0:12]]
gdb.head()
gdb.columns
gdb['intercept']=1
gdb[gdb.columns[[0,7]]].iloc[0:] #random but useful subsetting 
gdb.columns

####start of getting a df for k combos from n where ordering dos not matter and without replacement (n choose k) 
#use the expand grid function in the boostng and k fold cv file 
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}
#define x1 do be the values in xvars and x2 to be the values in xvar then use expand grid, then find the rows where the cols == the same value
#and repeated rearrangements
xvars=['Average Retail Price','Demo', 'Store','Endcap','Demo1-3', 'Sales Rep','Region']
#repeat xvars for k times in all TwoWay (figure out how to auto this make k an input then it repeats the xvar list k times in the grid function )
allTwoWay=pd.DataFrame(expandgrid(xvars,xvars,xvars,xvars)) 
#write a loop that goes through all the elements of xvars and rep them the len of ex above time extend a list with this and ,erge the list into 
#the df, then rep for 3...k xvars 
#get the unique values per row
rows=np.array(allTwoWay.index)
for i in range(0,len(allTwoWay)):
    rows[i]=len(allTwoWay.iloc[i].unique())
#gets the rows were there is no replacement 
allTwoWay=allTwoWay[rows==len(allTwoWay.columns)] #only keep the rows where the len unique of values in the row = the len of the cols 
allTwoWay.index=range(0,len(allTwoWay))
#now get the values for each ith row and sort them then make them as a string, then get the unique values of this as the predictors 
#combine the 2 as one element hoe to paste all the elemnets of a list in python: 
'_'.join(sorted(allTwoWay.iloc[0])) #loop through all the rows of allTwoway and do this 
twoxvars=list()
for i in range(0,len(allTwoWay)):
    twoxvars.append('_'.join(sorted(allTwoWay.iloc[i])))
#gets all possible combos of 2 x vars from the x var list 

#for the ith value in the array figure out how to split back into two elements, ie the vars to use in the df 
finalvars=list(range(0,len(np.unique(np.array(twoxvars))))) #goes through the length of the unique k var combos ordering does not matter 
for i in range(0,len(np.unique(np.array(twoxvars)))):
    finalvars[i]=np.unique(np.array(twoxvars))[i].split("_") #un pastes the vars 

for i in range(0,len(np.unique(np.array(twoxvars)))): #puts the intercept value in for each k combo of the x var names 
    finalvars[i].append('intercept')
finalvars
####end of getting a df for k combos from n where ordering dos not matter and without replacement (n choose k)

#fits an ols model for each combo of the unique k prodictors from the possible n x predictors 
rsquareds=list(range(0,len(finalvars)))
allregs=list()
for i in range(0,len(finalvars)):
    if(np.array(gdb[finalvars[i]].dtypes=='O').sum()>=1):
        a=pd.DataFrame(gdb[list(np.array(finalvars[i])[np.array(list(gdb[finalvars[i]].dtypes!='O'))])])
        b=pd.DataFrame(pd.get_dummies(gdb[list(np.array(finalvars[i])[np.array(list(gdb[finalvars[i]].dtypes=='O'))])]))
        est=sm.OLS(endog=gdb['Units Sold'], exog=pd.concat([a,b], axis=1), missing='drop').fit()
        allregs.append(est.summary())
        rsquareds[i]=est.rsquared_adj
    else:
        est=sm.OLS(endog=gdb['Units Sold'], exog=gdb[finalvars[i]], missing='drop').fit()
        allregs.append(est.summary())
        rsquareds[i]=est.rsquared_adj
rsquareds=np.array(rsquareds)
allregs[0]
len(rsquareds)
max(rsquareds)
np.where(rsquareds==max(rsquareds))
allregs[26]


