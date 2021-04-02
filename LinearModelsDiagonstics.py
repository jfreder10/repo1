import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels  import regression as reg
import statsmodels.api as sm
import plotly.express as px #if over so many obs then will not lot the points on the graph 
import itertools

cabs1=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\cabs.csv')
cabs1.head()
cabs=cabs1.sample(500)
cabs.columns
type(cabs)
cabs['intercept']=list(range(0,len(cabs)))
cabs['intercept']=1
cabs.head()
cabs=cabs.dropna() #rm the na values 
len(cabs)

#predict price from distance initialy then create diagonstics of the model 
modDist=sm.OLS(endog=cabs['price'], exog=cabs[['distance','intercept']]).fit()
modDist.summary()
dir(modDist) #gets all attribute functions of the sm.OLS object
#plot the residuals against the x values 
len(modDist.resid) 
cabs['distRes']=modDist.resid
distVariable=cabs['distance']
cabs.info()
fig=px.scatter(x=cabs['price'],y=cabs['distRes'])
fig.show() #problem with non equal variance try taking sqrt of the response and or var and re check 
#model with sqrt of y 
modDistYsqrt=sm.OLS(endog=cabs['price'] ** .5, exog=cabs[['distance','intercept']]).fit()
modDistYsqrt.summary()
fig=px.scatter(x=cabs['price'] ** 5,y=modDistYsqrt.resid) #defidently not as linear of a variance in the plot so better 
fig.show() #stil fixes some of the linear behavior of the residuals vs the actual y but only makes it like a downward quadradic
#fit poly function of x on the sqrt of y
cabs['distanceSquared']=cabs['distance'] ** 2
cabs.head()
modDistYsqrtDist2=sm.OLS(endog=cabs['price'] ** .5, exog=cabs[['distance','distanceSquared','intercept']]).fit()
modDistYsqrtDist2.summary()
fig=px.scatter(x=cabs['price'] ** 5,y=modDistYsqrtDist2.resid) #Some equal spread for large values of y but not as good as could be 
fig.show()

#try sqrt of x and if not better then try sqrt of both 
cabs['DistSqrt']=cabs['distance'] ** .5
modDistYsqrtX=sm.OLS(endog=cabs['price'] ** .5, exog=cabs[['DistSqrt','intercept']]).fit()
modDistYsqrt.summary()

#loop through all combos to raise x and y by with exp grid then find best r2
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

ypower=np.arange(.1, 5, .1)
len(ypower)
xpower=np.arange(.1, 5, .1)
len(xpower)
powers=pd.DataFrame(expandgrid(ypower,xpower))
len(powers)
rsqs=list(range(0,len(powers)))
powers.head()


cabs['DistPower']=cabs['distance'] ** powers['Var2'].iloc[0]
modDistYsqrtX=sm.OLS(endog=cabs['price'] ** powers['Var1'].iloc[0], exog=cabs[['DistPower','intercept']]).fit()
modDistYsqrt.summary()

for i in range(0, len(powers)):
    cabs['DistPower']=cabs['distance'] ** powers['Var2'].iloc[i]
    modDistYsqrtX=sm.OLS(endog=cabs['price'] ** powers['Var1'].iloc[i], exog=cabs[['DistPower','intercept']]).fit()
    rsqs[i]=modDistYsqrtX.rsquared

max(rsqs) #.18 find the value then row in the powers df to get the values to raise the values by  
arr=np.where(rsqs==max(rsqs)) #swts the best position to the var arr 
kthmod=arr[0].tolist()
powers.iloc[kthmod] #best powers to raise x and y to are 4.9 for y and 2.4 for x 



##special prokect DO NOT DO THIS UNTIL HAVE A LOT OF TIME!!!!!
#could loop through this as the inner loop while the outter would resample from the df n trials for each trial store the best combo to raise x and y to and the max 
#r^2 they correspond to 
##special prokect DO NOT DO THIS UNTIL HAVE A LOT OF TIME!!!!!

#mtcars (more numeric vars)
mtcars=pd.read_csv(r'C:\Users\fredr\Documents\StatTool\PythonCode\FunctionsAndCoolStuff\DATA\MTCARS.csv')
mtcars.head()
mtcars['intercept']=1
#predict mpg from hp then plot diagnostics and determine which other predictor should be included by plotting residuals to the other predictors not in model 
modMPGweight=sm.OLS(endog=mtcars['mpg'] , exog=mtcars[['wt','intercept']]).fit()
modMPGweight.summary() #high R^2 and highly sig 
#plot residuals 
modMPGweight.resid
fig=px.scatter(x=mtcars['wt'],y=modMPGweight.resid)  #not too bad but still somewhat of a trend 
fig.show()
#try log 
np.log10(mtcars['mpg'])
modLogMPGweight=sm.OLS(endog=np.log10(mtcars['mpg']) , exog=mtcars[['wt','intercept']]).fit()
modLogMPGweight.summary() #increased R^2
fig=px.scatter(x=mtcars['wt'],y=modLogMPGweight.resid) 
fig.show() #defidently better but a few leverage points 
#see if can recursively plot data in a loop with plotly express include fig and fig.show() in loop 
mtcars.head()
xvars1=['cyl','disp','hp','drat','qsec','carb']
fig=px.scatter(x=mtcars[xvars1[0]],y=modLogMPGweight.resid) 
fig.show()
#works, does open a tab for every xvar and shows the plots 
#make this into a function in the visulazations folder
for i in range(0,len(xvars1)):
    fig=px.scatter(x=mtcars[xvars1[i]],y=modLogMPGweight.resid) 
    fig.show()

#use function to get r2 for all combos then in a loop get the best transformation for all possible combos size k
def allRegCombos(df,yvar,xvars,k,met='rsquareds'):
    finalvars=list(itertools.combinations(xvars, k))
    for i in range(0,len(finalvars)):
        finalvars[i]=list(finalvars[i])
    rsquareds=list(range(0,len(finalvars)))
    allregs=list()
    for i in range(0,len(finalvars)):
        if(np.array(df[finalvars[i]].dtypes=='O').sum()>=1):
            a=pd.DataFrame(df[list(np.array(finalvars[i])[np.array(list(df[finalvars[i]].dtypes!='O'))])])
            b=pd.DataFrame(pd.get_dummies(df[list(np.array(finalvars[i])[np.array(list(df[finalvars[i]].dtypes=='O'))])]))
            est=sm.OLS(endog=df[yvar], exog=pd.concat([a,b], axis=1), missing='drop').fit()
            allregs.append(est.summary())
            rsquareds[i]=est.rsquared_adj
        else:
            est=sm.OLS(endog=df[yvar], exog=df[finalvars[i]], missing='drop').fit()
            allregs.append(est.summary())
            rsquareds[i]=est.rsquared_adj

    if(met=='rsquareds'):
        return rsquareds
    else:
        return allregs
xvars1=['cyl','disp','hp','drat','qsec','carb']
y='mpg'
allRegCombos(df=mtcars,yvar=y,xvars=xvars1,k=2,met='allregs')
#make function to consider all two way interactions among any set of vars selected 





