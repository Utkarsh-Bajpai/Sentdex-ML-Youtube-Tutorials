import numpy as np
from sklearn import cross_validation, svm
import pandas as pd

df = pd.read_csv('data.csv')
#Replace missing data, outlier
df.replace('?',-99999,inplace=True)

#Remove irrelevant data
df.drop(['id'],1,inplace = True)

X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])

#Split it into a Validation test set and training set
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)

#Define the classifier, in this case SVM (SVC)
clf = svm.SVC()
clf.fit(X_train,Y_train)

#Print the accuracy
accuracy = clf.score(X_test,Y_test)
print(accuracy)