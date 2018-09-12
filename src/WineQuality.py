#Step 1: Import necessary packagesimport math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as numpy
from numpy.linalg import norm, lstsq
import math

#step 2: Get data from the file and encode the labels using LabelEncoder class.
Data= pd.read_table("/Users/llu159/Desktop/small_code/classification/Wine_Quality_Data_Set/winequality-red.csv", sep= None, engine= "python")

#step 3: split the data into training and testing set.
data_train, data_test= train_test_split(Data, test_size= 0.2, random_state= 42)
X_train= data_train.drop("quality", axis=1)
print(X_train.shape)
Y_train= data_train["quality"]
print(Y_train.shape)
X_test= data_test.drop("quality", axis=1)
print(X_test.shape)
Y_test= data_test["quality"]
print(Y_test.shape)


#step 4: Scale the data using StandardScaler class.
scaler= StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)


#step 5: Determine centers of the neurons using KMeans.
K_cent= 8
km= KMeans(n_clusters= K_cent, max_iter= 100)
km.fit(X_train)
cent= km.cluster_centers_


#step 6: Determine the value of [latex]\sigma[/latex]
max=0
for i in range(K_cent):
	for j in range(K_cent):
		d= numpy.linalg.norm(cent[i]-cent[j])
		if(d> max):
			max= d
d= max
sigma= d/math.sqrt(2*K_cent)
print('The sigma is:',sigma)

#step 7: Set up matrix G, the outpu of hidden layer neuron.
shape= X_train.shape
row= shape[0]
print('The row is:',row)
print('The X_train[0] is:',X_train[0])
column= K_cent
print('The column is:',column)
G= numpy.empty((row,column), dtype= float)
for i in range(row):
    for j in range(column):
        dist= numpy.linalg.norm(X_train[i]-cent[j])
        G[i][j]= math.exp(-norm(dist)/(2*math.pow(sigma,2)))

#step 8: Find weight matrix W to train the network.
GTG= numpy.dot(G.T,G)
GTG_inv= numpy.linalg.inv(GTG)
fac= numpy.dot(GTG_inv,G.T)
W= numpy.dot(fac,Y_train)


#step 9: Set up matrix G for the test set.
row = X_test.shape[0]
column = K_cent
G_test = numpy.empty((row, column), dtype=float)
for i in range(row):
    for j in range(column):
        dist = numpy.linalg.norm(X_test[i] - cent[j])
        G_test[i][j] = math.exp(-norm(dist) / (2*math.pow(sigma, 2)))

#step 10: Analyze the accuracy of prediction on test set
prediction= numpy.dot(G_test,W)
prediction= 0.5*(numpy.sign(prediction-0.5)+1)
print(prediction)


score= accuracy_score(prediction,Y_test)
print ('accuracy:',score)
print ('The accuracy of prediction on test set:',score.mean())


