import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels  import regression as reg
import statsmodels.api as sm
import plotly.express as px
heart=pd.read_csv(r"C:\Users\fredr\Documents\StatTool\PythonCode\heart.csv")
#fit models with transformed x values to y then plot the y hat of these as y on a plot and original xas x to see better relationshipbetween x and y 
#loop through each pairwise combo of the vars and the get the cor, find not that good of ones and transform x to increase correlation 
indvars=['age','sex','cp','trestbps','chol']
k=len(indvars)
xvarcors=pd.DataFrame(np.zeros((k,k)))
sm.OLS(endog=heart[indvars[0]], exog=heart[indvars[1]], missing='drop').fit().rsquared

for i in range(0,len(xvarcors)):
    for j in range(0,len(xvarcors)):
        xvarcors.iloc[i,j]=sm.OLS(endog=heart[indvars[i]], exog=heart[indvars[j]], missing='drop').fit().rsquared

xvarcors
#increase the cor between the 3rd and 4th vars with transformation, R^2=0.466402  with no transformation 
#loop through many possible x var transformations andfind one with hifhest R^2
x=heart[indvars[4]]
y=heart[indvars[3]]
px.scatter(x=x,y=y).show()


#make a seq from .5 to 4 by .5 to loop through, raise x to this and fit mod to y and get R^2
lambdas=np.arange(.5, 4, .5)
transrsq=list(range(0,len(lambdas)))
for i in range(0,len(transrsq)):
    transrsq[i]=sm.OLS(endog=y, exog=x**lambdas[i], missing='drop').fit().rsquared
transrsq
np.where(transrsq==max(transrsq))
#get the predicted values for x**lambdas[1]
x**lambdas[0]
yhat=sm.OLS(endog=y, exog=x**lambdas[0], missing='drop').fit().predict(x)



#fitting x transformed to original y (actual y not fitted values)
fig=px.scatter(x=x**lambdas[5],y=y) #
fig.show()
#fitting x transformed to fitted y (predicted y not actual values)
fig1=px.scatter(x=x**lambdas[5],y=yhat) #nonlinear relationship, predicting y from x with a nonlinear function 
fig1.show()
#fitting x original to original y values 
fig=px.scatter(x=x,y=y) 
fig.show()
#fitting x original to the y hat values 
fig=px.scatter(x=x,y=yhat) 
fig.show()

#make a svm with sklearn to predict target, plot the margin with and without a projection










