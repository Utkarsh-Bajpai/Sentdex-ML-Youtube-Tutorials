import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[2,3]])

#Scatter the data points for demonstration
#plt.scatter(X[:,0], X[:,1], s = 100)
#plt.show()

#Define the classifier
clf = KMeans(n_clusters=2)
clf.fit(X)

centers = clf.cluster_centers_
labels = clf.labels_
#But there are only 2 classes
colors = ["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(centers[:,0], centers[:,1], marker='x', s = 150, linewidths=5)
plt.show()


