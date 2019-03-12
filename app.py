import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

#duomenys
x = [1, 5, 1.5, 8, 1, 9] 
y = [2, 8, 1.8, 8, 0.6, 11]
plt.scatter(x, y)
plt.show()

#duomenu convertavimas i array
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
#isreiskiama kmeans ir initialiiznam ji su reikiamu clusteriu kiekiu
kmeans = KMeans(n_clusters = 2)
#duomenu leidimas per kmeans
kmeans.fit(X)
#centru ir leibeliu isreikimas kintamaisiais
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
#output 
print(centroids)
print(labels)


#duomenu viusalizacija
colors = ["g.","r.","c.","y."]
#loop kuris grupuoja pagal cluster ceroidus
for i in range(len(x)):
    #duomenu atvaizdavimas
     print("coordinate: ",X[i],"label: ",labels[i])
     plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(centroids[:,0],centroids[:,1],marker = "x", s=150, linewidths = 5, zorder = 10)  
#grafikas
plt.show()
