# Knn using scikit-learn and own data set.

import numpy as np
import pandas as pd
from sklearn import neighbors
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv('Data_sets/coords_ABC.csv', sep=';', decimal=',')

X = df[['x', 'y']].values
y = df['class'].values

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#e0ecf4', '#9ebcda', '#8856a7'])
cmap_bold = ListedColormap(['#af8dc3', '#f7f7f7', '#7fbf7b'])

n_neighbors = 3

# Create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % n_neighbors)

plt.show()

p = np.array([0, 0.5])
clf.predict([p])
