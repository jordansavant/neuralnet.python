import perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# rgen = np.random.RandomState()
# n = rgen.normal(loc=0.0, scale=1, size=1)
# print(n)

p = perceptron.Perceptron()
# print(p)

# a = [1, 2, 3]
# b = [4, 5, 6, 7, 8]
# c = (9, 10, 11, 12)
# d = {"a": "a", "b": "b"}

# for w, x, y, z in zip(a, b, c, d):
#     print("w", w)
#     print(" x", x)
#     print("  y", y)
#     print("   z", z)

# Get a CSV of the iris data set, returns a pandas DataFrame
# DataFrame is a 2d data structure with labeld axes
df = pd.read_csv("iris.data.csv", header=None)

# Pull out first 100 rows of data which is 50 setosa flowers and 50 versicolor flowers
# Set setosa = -1 and versicolor = 1 so we can classify them
# Also take col 1 (sepal length) and col 3 (petal length) and assign to feature matrix X

# get subset of first 100 rows and fourth column
y = df.iloc[0:100, 4].values
# reassign class label to -1 or 1
y = np.where(y == 'Iris-setosa', -1, 1)

# get sepal length and petal length cols
X = df.iloc[0:100, [0, 2]].values

# plot the data in a visual way
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()

