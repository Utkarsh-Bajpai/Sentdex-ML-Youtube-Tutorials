import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies = []

for i in range(25):
    df = pd.read_csv('data.csv')
    #Replace missing data, outlier
    df.replace('?',-99999,inplace=True)

    #Remove irrelevant data
    df.drop(['id'],1,inplace = True)

    X = np.array(df.drop(['class'],1))
    Y = np.array(df['class'])

    #Split it into a Validation test set and training set
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)

    #Define the classifier
    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train,Y_train)

    #Print the accuracy
    accuracy = clf.score(X_test,Y_test)
    accuracies.append(accuracy)
print(sum(accuracies) / len(accuracies))

    #example = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
    #example = example.reshape(len(example),-1)
    #prediction = clf.predict(example)
    #print(prediction)