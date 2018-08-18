from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#Define some random data
#X = np.array([1,2,3,4,5,6],dtype=np.float64)
#Y = np.array([5,4,6,5,6,7],dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    Y = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        Y.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    X = [i for i in range(len(Y))]
    return np.array(X,dtype=np.float64), np.array(Y,dtype=np.float64)

X,Y = create_dataset(40,40,2,correlation='pos')

#Calculate the slope and intercept with the y axis from the data
def best_fit_slope_intercept(X,Y):
    m = (((mean(X) * mean(Y)) - mean(X * Y)) /
         (mean(X) * mean(X) - mean(X ** 2)))
    b = mean(Y) - m * mean(X)
    return m,b

#Calculate the squared error
def squared_error(Y_orig, Y_est):
    return sum((Y_est-Y_orig)**2)

#How accurate is the best fit line?
def c_determination(Y_orig, Y_est):
    Y_mean = np.array([mean(Y_orig) for y in Y_orig], dtype=np.float64)
    squared_error_regr = squared_error(Y_orig, Y_est)
    squared_error_Y_mean = squared_error(Y_orig, Y_mean)
    return 1 - (squared_error_regr/squared_error_Y_mean)

m,b = best_fit_slope_intercept(X,Y)
line = np.array([(m*x) + b for x in X],dtype=np.float64)
c = c_determination(Y, line)
print(c)

#Show the the line and the data
plt.scatter(X,Y)
plt.plot(X,line)
plt.show()


