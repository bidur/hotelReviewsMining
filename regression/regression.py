#https://datatofish.com/multiple-linear-regression-python/
#https://datatofish.com/statsmodels-linear-regression/

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import linear_model
import statsmodels.api as sm #http://www.statsmodels.org/devel/_modules/statsmodels/regression/linear_model.html#RegressionResults
import seaborn as sn
import numpy as np
np.set_printoptions(suppress=True) # to avoid printing number ins scientific format





def checkLinearity( dependent, independent):
    plt.scatter(df[independent].astype(float), df[dependent].astype(float), color='red')
    plt.title(dependent+' Vs '+ independent, fontsize=14)
    plt.xlabel(independent, fontsize=14)
    plt.ylabel(dependent, fontsize=14)
    plt.grid(True)
    plt.show()
    

hotelData = pd.read_csv('_2X_3gram_ALL_HOTEL_TOPIC_VALUES_0_r.csv')
df = DataFrame(hotelData,columns=['hotel','Rating','totalScore','Asp_0','Asp_1','Asp_2','Asp_3','Asp_4','Asp_5','Asp_6','Asp_7','Asp_8','Asp_9','Asp_10','Asp_11','Asp_12','Asp_13','Asp_14','Asp_15','Asp_16','Asp_17','Asp_18','Asp_19'])
df = df.dropna()
df2= df
df2.drop(['hotel','Rating'], axis=1, inplace=True)


f= plt.subplots(figsize=(29,21))
sn.heatmap(df2.corr(),annot=True,fmt='.1f',color='green')
#plt.show()
plt.savefig('_2X_3gram_reviews0-200.png', bbox_inches='tight')




aspectList =[]
for i in range (20):
    aspect = 'Asp_'+str(i)
    #checkLinearity( 'totalScore', aspect )
    aspectList.append( aspect )



X = df[aspectList].astype(float) # here we have all 20 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Asp_0'] for example.Alternatively, you may add additional variables in the list
Y = df['totalScore'].astype(float)

######################################################################################


testCount = 40#40

# Split the data into training/testing sets
X_train = X[:-testCount]
X_test = X[-testCount:]

# Split the targets into training/testing sets
y_train = Y[:-testCount]
y_test = Y[-testCount:]

# with sklearn
regr = linear_model.LinearRegression(fit_intercept =True)
regr.fit(X_train, y_train)


# Make predictions using the testing set
y_pred = regr.predict(X_test)

print('Intercept: ', np.around(regr.intercept_,decimals=4))

print('Coefficients: ', np.around(regr.coef_,decimals=2))


print("Mean squared error: %.2f"  % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))



exit()

# with statsmodels
X = sm.add_constant(X) # adding a constant # It tells the model to fit a value for b as well as coefficients for your predictors. 

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) 
  
print_model = model.summary()
print model.params
print(print_model)


''' OUTPUT DESCRIPTION ( print(print_model))
Adjusted. R-squared reflects the fit of the model. R-squared values range from 0 to 1, where a higher value generally indicates a better fit, assuming certain conditions are met.
const coefficient is your Y-intercept. It means that if both the Interest_Rate and Unemployment_Rate coefficients are zero, then the expected output (i.e., the Y) would be equal to the const coefficient.
Interest_Rate coefficient represents the change in the output Y due to a change of one unit in the interest rate (everything else held constant)
Unemployment_Rate coefficient represents the change in the output Y due to a change of one unit in the unemployment rate (everything else held constant)
std err reflects the level of accuracy of the coefficients. The lower it is, the higher is the level of accuracy
P >|t| is your p-value. A p-value of less than 0.05 is statistically significant
Confidence Interval represents the range in which our coefficients are likely to fall (with a likelihood of 95%)

MOre: http://connor-johnson.com/2014/02/18/linear-regression-with-python/
'''
