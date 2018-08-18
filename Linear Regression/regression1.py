'''
Use linear regression to predict stock prices
'''

import scipy
import pandas as pd
import time
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import os.path

style.use('ggplot')

quandl.ApiConfig.api_key = 'XvbS-WQy2LhgsF1K2km-'

#Get the stock trade info
df  = quandl.get_table('WIKI/PRICES')
print(df.head())
df['HL_PCT'] =(df['adj_high']-df['adj_close'])/df['adj_close'] * 100.0
df['PCT_change'] =(df['adj_close']-df['adj_open']) / df['adj_open'] * 100.0
df = df[['adj_close','HL_PCT','PCT_change','adj_volume']]
forecast_col = df['adj_close']
forecast_out = int(math.ceil(0.1*len(forecast_col)))
df['label'] = forecast_col.shift(-forecast_out)
df.dropna(inplace=True)

X=np.array(df.drop(['label'],1))
X=preprocessing.scale(X)
X_lately = X[:-forecast_out]
X=X[:-forecast_out]

Y=np.array(df['label'])
Y=Y[:-forecast_out]

#Create validation and training sets
x_train,x_test,y_train,y_test=cross_validation.train_test_split(X,Y, test_size=0.2)
clf=LinearRegression(n_jobs=-1)

#Save the classifier without the necessity to retrain
if os.path.isfile('linearregression.pickle') is not True:
    clf.fit(x_train, y_train)
    with open('linearregression.pickle', 'wb') as f:
        pickle.dump(clf,f)
else:
    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)

#Write out the accuracy
accuracy=clf.score(x_test,y_test)
forecast_set=clf.predict(X_lately)
print(accuracy,forecast_set)
forecast_set=np.array(forecast_set)
df['Forecast'] = np.nan

#Arrange the data so it can be graphed
last_date=df.iloc[-1].name
last_unix=last_date
one_day=1
next_unix=last_unix+one_day

for i in forecast_set:
 next_date = next_unix
 next_unix+=one_day
 df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]

print(df.label[:-forecast_out])
print(forecast_set)
df['adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#SVM
'''
clf_svm = svm.SVR()
#Train
clf_svm.fit(X_train, y_train)

#Accuracy of learning
accuracy_svm = clf_svm.score(X_test,y_test)
print(accuracy_svm)
'''


