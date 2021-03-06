# *************************************
# PCA SCRIPT MATH 140
# Copyright of Kimi Holsapple
# 
# Analyzes small input data set from Problem 7, HW 2 and performs PCA 
# Outputs PCA and Graphs
# *************************************

import math
import numpy as np
import statistics as stats
from numpy.linalg import eig
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt

#*************************************************

def standardize (data):
	n = len(data)
	data_stdev = stats.stdev(data)
	data_mean = stats.mean(data)
	clean_data = []
	for i in data:
		clean_data.append( (i - data_mean)/data_stdev )
	return clean_data

def standardize_nostdev (data):
	n = len(data)
	# data_stdev = stats.stdev(data)
	data_mean = stats.mean(data)
	clean_data = []
	for i in data:
		clean_data.append( (i - data_mean) )
	return clean_data


#*************************************************


#*************************************************
# HArdcoded implemenation using pca library scikit 

#*************************************************

# src @ https://medium.com/analytics-vidhya/understanding-principle-component-analysis-pca-step-by-step-e7a4bb4031d9
# src @ https://plotly.com/python/pca-visualization/
# Using lib imports
print("Python-Calc")
n_components = 1
print("Our data set")
# data_raw = np.matrix( [[-1, 1, 0, 2], 
# 					   [-2, 3, 1, 0.5]])
# HARD coded 
data_raw = np.matrix( [ [-1, -2],
						[1,3], 
						[0, 1],
						[2, 0.5]
						])

data = pd.DataFrame(data_raw, columns = ['C1','C2'])
print(data)
data_X = standardize([-1, 1, 0, 2])
data_Y = standardize([-2, 3, 1, 0.5])
data_std = np.asarray( [ [data_X[0], data_Y[0]],
						[data_X[1], data_Y[1]], 
						[data_X[2], data_Y[2]],
						[data_X[3], data_Y[3]]
						])
# data_std = (data - data.mean())/ data.std()
print("standardized data")
print(data_std) 
pca = PCA(n_components = n_components)
princ_comp = pca.fit_transform(data_std)
princ_data = pd.DataFrame(data=princ_comp, columns=['nf'+str(i+1) for i in range(n_components)])
print("pca data")
print(princ_data)







# ***************************
# "By Hand"
print('\n')
print("*********************")
print ("By-Hand Calculations")

data_set_X = [-1, 1, 0, 2]
data_set_Y = [-2, 3, 1, 0.5]

# standardize the data set

print ("Step 1: standardize ")
data_X = standardize(data_set_X)
data_Y = standardize(data_set_Y)
data = np.asarray( [ [data_X[0], data_Y[0]],
						[data_X[1], data_Y[1]], 
						[data_X[2], data_Y[2]],
						[data_X[3], data_Y[3]]
						])
print(data)

# calculate the covariance 4x4 since onlt f1 f2
print("*********************")
print ("Step 2: covariance ")

covMatrix = np.cov(data_X, data_Y)
print (covMatrix)

# calculate th eigenvaules and eigenvectors
print("*********************")
print ("Step 3: eigenvaules and eigenvectors ")
eigen = np.array(covMatrix)
w,v=eig(eigen)
print('E-value:', w)
print('E-vector \n', np.asarray(v))

# # Sort them to corresponding eigenvectors
# print("*********************")
# print("Step 4. picking first column of e1 eigenvectors: ")
# eigen_mat = [v[0][0], v[1][0]]
# print( eigen_mat )

# pick k eigenvaules and form a matrix of eigenvectors
print("*********************")
print("Step 4. k eigen with k = 1 and now mutiply")

# Transform the original matrix
# transform = np.dot(data, eigen_mat )
# print(transform)
transform = np.dot(data, v )
print(transform)

print("*********************")
print("Result")
print("The PCA explained variance ratio is: ")
print(pca.explained_variance_ratio_)


###### GRAPH /////// TODO

X_new = pca.inverse_transform(princ_comp)
X = data_std
plt.title("PCA Graph")

v_x = [v[0][0], v[0][1]]
v_y = [v[1][0], v[1][1]]
print(v_x)
print(v_y)

t_x = [transform[0][0], transform[1][0], transform[2][0], transform[3][0] ] 
t_y = [transform[0][1] ,transform[1][1] , transform[2][1] , transform[3][1]]

plt.scatter(v_x,  v_y, alpha=0.6)				 
v_x = [v[0][0], 0]
v_y = [v[1][0], 0]
plt.plot(v_x,  v_y, alpha=0.6)

v_x = [0,  v[0][1]]
v_y = [0, v[1][1]]	
plt.plot(v_x,  v_y, alpha=0.6)


# plt.plot(transform, transform, alpha=0.8)		 
# plt.scatter(w[0], w[1], alpha=0.8)						# eigen vaules
# plt.plot(princ_comp, princ_comp, alpha=0.8)
plt.scatter(data_set_X, data_set_Y, alpha=0.2)	
plt.scatter(t_x, t_y, alpha=0.8)		
plt.axis('equal');
# plt.xlabel("Principal Component 1 ",fontsize=14)
# plt.ylabel("Principal Component 1 ",fontsize=14)
plt.show()


# plt.scatter(data_set_X, data_set_Y, alpha=0.2)	
# pca.fit(data) 

# X_pca=pca.transform(data) 

# plt.scatter(data_set_X, data_set_Y, alpha=0.2)
# plt.scatter(X_pca, X_pca, alpha=0.2)
# plt.scatter(w[0], w[0], alpha=0.8)
# plt.scatter(w[1], w[1], alpha=0.8)
# plt.xlabel("Principal Component 1 ",fontsize=14)
# plt.show()

# PC_values = np.arange(pca.n_components) + 1
# plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Variance Explained')
# plt.show()
