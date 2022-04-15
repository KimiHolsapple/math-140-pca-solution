# KNN algorithm for problem 3
# copyright Kimi Holsapple

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
import math


Point_dict = {
				"[2, 10]" : "A1",
			  	"[2, 5]" : "A2" ,
			  	"[8, 4]" : "A3",
			  	"[5, 8]" : "A4",
			  	"[7, 5]" : "A5" ,
			  	"[6, 4]" : "A6",
			  	"[1, 2]" : "A7",
			  	"[4, 9]": "A8" 
			 }

g_threshold = 4
euc_dis_mat = []
neighbor_list = []

def show_nn(arr, point):
	neighbours = []
	print ("Point " + Point_dict[ str(point) ]  + " Nearest neighbours Are :" )
	for i in arr:
		if i[0] != 0 and i[0] <= g_threshold :
			print(Point_dict[ str(i[1]) ])
			neighbours.append(i[1] )
		#ignore == 0 since that is itself
	return neighbours

def euclidean_distance_formula(p1, p2):
	return math.sqrt( pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2) )

def euclidean_distance(p1, data):

	array_distances = []
	euc_dis_ref = []
	for p2 in data:

		if p1 == p2:
			array_distances.append([0.0, p2])
			euc_dis_ref.append(0.0)

		else:
			array_distances.append( [round( euclidean_distance_formula( p1, p2), 2 ), p2])
			euc_dis_ref.append(round( euclidean_distance_formula( p1, p2), 2 ))


	euc_dis_mat.append(euc_dis_ref)
	return array_distances


print("***********K Nearest Neighbor Clustering Algorithm*********** \n")

data = [ [2,10], [2,5], [8,4], [5,8], [7,5], [6,4], [1,2], [4,9]]


distance_key_set = []
for i in range(len(data)):
	distance_key_set.append( euclidean_distance(data[i], data) )

# Src @ https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/


print("***********Obtain Euclidean distances*********** \n")
for i in range(len(data)):		
	print("A_" + str(i+1) + ":" + str(euc_dis_mat[i]))

# Obtain nearest neighbours
print("\n***********Obtain nearest neighbours*********** \n")
print(" Sorted distances cutoff is threshold = 4 \n")
for i in range(len(data)):   #row
	for j in range(len(data)):		#column
			distance_key_set[i].sort(key=lambda tup: tup[0])


for i in range(len(distance_key_set)):		
	print("A_" + str(i+1) + ":" + str(distance_key_set[i]))

print("\n Display Nearest Nieghbour for each point \n")
for i in range(len(distance_key_set)):	
	neighbor_list.append(show_nn(distance_key_set[i], distance_key_set[i][0][1]))

print("\nMatrix \n")

statistics = []
statistics_internal = [0]*8
count = 1
for neighbor in neighbor_list:
	for i in range(len(neighbor)):
		key = Point_dict[ str(neighbor[i]) ]
		if key == "A1":
				statistics_internal[0] += 1
		if key == "A2":
				statistics_internal[1] += 1
		if key == "A3":
				statistics_internal[2] += 1
		if key == "A4":
				statistics_internal[3] += 1
		if key == "A5":
				statistics_internal[4] += 1
		if key == "A6":
				statistics_internal[5] += 1
		if key == "A7":
				statistics_internal[6] += 1
		if key == "A8":
				statistics_internal[7] += 1
	statistics.append(statistics_internal)
	statistics_internal = [0]*8
print("      A1 A2 A3 A4 A5 A6 A7 A8")
for i in statistics:
	print("A_" + str(count) + ": " + str(i))
	count += 1


print("\n***********Grahphical represnetation*********** \n")
labels = ["A_1", "A_2", "A_3", "A_4", "A_5", "A_6", "A_7", "A_8"]
x = []
y = []

x_1 = [2, 5, 4]
y_1 = [10, 8, 9]

x_2 = [2, 1]
y_2 = [5, 2]

x_3 = [8, 7, 6]
y_3 = [4, 5, 4]
for i in data:
	x.append(i[0])
	y.append(i[1])

plt.scatter(x, y)
plt.scatter(x_1, y_1)
plt.scatter(x_2, y_2)
plt.scatter(x_3, y_3)
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]))

# count = 0
# for neighbor in neighbor_list:
# 	for i in range(len(neighbor)):
# 		x = [ data[count][0], neighbor[i][0] ]
# 		y = [ data[count][1], neighbor[i][1] ]
# 		plt.plot(x, y, alpha=0.5)
# 	#print(neighbor)
# 	count += 1

plt.show()


