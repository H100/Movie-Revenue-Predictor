import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import sys
from numpy import *
import csv
from collections import Counter
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model

#For Classification
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier



genre = ['Action','Adventure','Fantasy','Sci-Fi','Thriller','Romance','Animation','Comedy','Family','Fantasy','Musical','Mystery','Western','Drama','Sport','Crime','Horror','History','War','Biography']

total = 3770


col0 = []#color/bw
col1 = []#director name
col6 = []#actor_2_name
col9 = []#genre
col10 = []#actor_10_name
col13 = []#actor_3_name
col16 = []#language
col17 = []#country
col18 = []#content_rating
cf9 = []
cf10 = []
cf11 = []
cf12 = []
cf13 = []
cf14 = []
cf15 = []
cf16 = []
cf17 = []
cf18 = []
cf19 = []
cf20 = []
cf21 = []
cf22 = []
cf23 = []
cf24 = []


with open('imdb_movie_dataset.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		if '' in row:
			continue
		col0.append(row[0])
		col1.append(row[1])

		cf9.append(row[2])
		cf10.append(row[3])
		cf11.append(row[4])
		cf12.append(row[5])

		col6.append(row[6])

		cf13.append(row[7])
		cf14.append(row[8])#Revenue

		col9.append(row[9])
		col10.append(row[10])

		cf15.append(row[11])
		cf16.append(row[12])

		col13.append(row[13])

		cf17.append(row[14])
		cf18.append(row[15])

		col16.append(row[16])
		col17.append(row[17])
		col18.append(row[18])

		cf19.append(row[19])
		cf20.append(row[20])
		cf21.append(row[21])
		cf22.append(row[22])
		cf23.append(row[23])
		cf24.append(row[24])


#Genre work
genre_1_hot = zeros((20,3770))
count=-1

for ele in col9:
	i=-1
	count += 1
	for gen in genre:
		i += 1
		if gen in ele:
			genre_1_hot[i][count]=1


genre_1_hot = np.array(genre_1_hot)

#print(genre_1_hot)

col0_dict = Counter(col0)
col1_dict = Counter(col1)
col6_dict = Counter(col6)
#col9_dict = Counter(col9)
col10_dict = Counter(col10)
col13_dict = Counter(col13)
col16_dict = Counter(col16)
col17_dict = Counter(col17)
col18_dict = Counter(col18)

count=1
for key in col0_dict:
	col0_dict[key] = count
	count += 1
count=1
for key in col1_dict:
	col1_dict[key] = count
	count += 1
count=1
for key in col6_dict:
	col6_dict[key] = count
	count += 1
count=1
for key in col10_dict:
	col10_dict[key] = count
	count += 1
count=1
for key in col13_dict:
	col13_dict[key] = count
	count += 1
count=1
for key in col16_dict:
	col16_dict[key] = count
	count += 1
count=1
for key in col17_dict:
	col17_dict[key] = count
	count += 1
count=1
for key in col18_dict:
	col18_dict[key] = count
	count += 1

cf0 = []
cf1 = []
cf2 = []
cf3 = np.transpose(genre_1_hot)
cf4 = []
cf5 = []
cf6 = []
cf7 = []
cf8 = []


for ele in col0:
	for key in col0_dict:
		if ele == key:
			cf0.append(col0_dict[key])
			

for ele in col1:
	for key in col1_dict:
		if ele == key:
			cf1.append(col1_dict[key])
			

for ele in col6:
	for key in col6_dict:
		if ele == key:
			cf2.append(col6_dict[key])
			

for ele in col10:
	for key in col10_dict:
		if ele == key:
			cf4.append(col10_dict[key])
			

for ele in col13:
	for key in col13_dict:
		if ele == key:
			cf5.append(col13_dict[key])
			

for ele in col16:
	for key in col16_dict:
		if ele == key:
			cf6.append(col16_dict[key])
			

for ele in col17:
	for key in col17_dict:
		if ele == key:
			cf7.append(col17_dict[key])
			

for ele in col18:
	for key in col18_dict:
		if ele == key:
			cf8.append(col18_dict[key])
			


cf0 = np.array(cf0).astype(np.float)
cf1 = np.array(cf1).astype(np.float)
cf2 = np.array(cf2).astype(np.float)
cf3 = np.array(cf3).astype(np.float)
cf4 = np.array(cf4).astype(np.float)
cf5 = np.array(cf5).astype(np.float)
cf6 = np.array(cf6).astype(np.float)
cf7 = np.array(cf7).astype(np.float)
cf8 = np.array(cf8).astype(np.float)
cf9 = np.array(cf9).astype(np.float)
cf10 = np.array(cf10).astype(np.float)
cf11 = np.array(cf11).astype(np.float)
cf12 = np.array(cf12).astype(np.float)
cf13 = np.array(cf13).astype(np.float)
cf14 = np.array(cf14).astype(np.float)#revenue
cf15 = np.array(cf15).astype(np.float)
cf16 = np.array(cf16).astype(np.float)
cf17 = np.array(cf17).astype(np.float)
cf18 = np.array(cf18).astype(np.float)
cf19 = np.array(cf19).astype(np.float)
cf20 = np.array(cf20).astype(np.float)
cf21 = np.array(cf21).astype(np.float)
cf22 = np.array(cf22).astype(np.float)
cf23 = np.array(cf23).astype(np.float)
cf24 = np.array(cf24).astype(np.float)



#print(int(cf24[2])/10)
X_not_3_14_22 = np.column_stack((cf0,cf1,cf2,cf4,cf5,cf6,cf7,cf8,cf9,cf10,cf11,cf12,cf13,cf15,cf16,cf17,cf18,cf19,cf20,cf21,cf23,cf24))
"""
X_rate = np.column_stack((np.column_stack((X_not_3_14_22,cf14)),np.power(np.column_stack((X_not_3_14_22,cf14)),2),np.power(np.column_stack((X_not_3_14_22,cf14)),3),cf3))
"""
X_rate = np.column_stack((np.column_stack((X_not_3_14_22,cf14)),cf3))
Y_rate = cf22
a = np.column_stack((X_rate,Y_rate))
np.savetxt("rating.csv", a, delimiter=",")
"""
X_rev = np.column_stack((np.column_stack((X_not_3_14_22,cf22)),np.power(np.column_stack((X_not_3_14_22,cf22)),2),np.power(np.column_stack((X_not_3_14_22,cf22)),3),cf3))
"""
X_rev = np.column_stack((np.column_stack((X_not_3_14_22,cf22)),cf3))
Y_rev = cf14
b = np.column_stack((X_rev,Y_rev))
np.savetxt("revenue.csv", b, delimiter=",")


np.resize(X_rate,(3770,43))
np.resize(Y_rate,(3770,1))

X_train_rate, X_test_rate, Y_train_rate, Y_test_rate = train_test_split(X_rate, Y_rate, test_size=0.33, random_state=42)
"""

#==================================================================RATING====================================================================
print("Rating")

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
y_lin = regr.fit(X_train_rate, Y_train_rate).predict(X_test_rate)
print(np.mean(abs(y_lin-Y_test_rate)**2)**0.5)

np.savetxt("rating_linear.csv", y_lin, delimiter=",",fmt='%.1e')
print("LR is over")



svr_rbf = SVR(kernel='rbf', C=0.1, gamma=0.2)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X_train_rate, Y_train_rate).predict(X_test_rate)
print(np.mean(abs(y_rbf-Y_test_rate)**2)**0.5)
np.savetxt("rating_rbf.csv", y_rbf, delimiter=",",fmt='%.1e')
print("RBF is over")



krr = KernelRidge(alpha=0.1)
y_krr = krr.fit(X_train_rate, Y_train_rate).predict(X_test_rate)
print(np.mean(abs(y_krr-Y_test_rate)**2)**0.5)
np.savetxt("rating_krr.csv", y_krr, delimiter=",",fmt='%.1e')
print("KRR is over")


lasso = linear_model.Lasso(alpha=0.1)
y_lasso = lasso.fit(X_train_rate, Y_train_rate).predict(X_test_rate)
print(np.mean(abs(y_lasso-Y_test_rate)**2)**0.5)
np.savetxt("rating_lasso.csv", y_lasso, delimiter=",",fmt='%.1e')
print("LASSO is over")
#==================================================================REVENUE====================================================================
print("Revenue")
np.resize(X_rev,(3770,43))
np.resize(Y_rev,(3770,1))
#print(X.shape)
#print(Y.shape)

X_train_rev, X_test_rev, Y_train_rev, Y_test_rev = train_test_split(X_rev, Y_rev, test_size=0.2, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
y_lin = regr.fit(X_train_rev, Y_train_rev).predict(X_test_rev)

print(np.mean(abs(y_lin-Y_test_rev)**2)**0.5)

np.savetxt("revenue_linear.csv", y_lin, delimiter=",")
print("LR is over")



svr_rbf_rev = SVR(kernel='rbf', C=0.1, gamma=0.2)
svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf_rev = svr_rbf_rev.fit(X_train_rev, Y_train_rev).predict(X_test_rev)
print(np.mean(abs(y_rbf_rev-Y_test_rev)**2)**0.5)
np.savetxt("revenue_rbf.csv", y_rbf_rev, delimiter=",")
print("RBF is over")


krr = KernelRidge(alpha=0.1)
y_krr = krr.fit(X_train_rev, Y_train_rev).predict(X_test_rev)
print(np.mean(abs(y_krr-Y_test_rev)**2)**0.5)
np.savetxt("revenue_krr.csv",y_krr , delimiter=",")
print("KRR is over")


lasso = linear_model.Lasso(alpha=0.1)
y_lasso = lasso.fit(X_train_rev, Y_train_rev).predict(X_test_rev)
print(np.mean(abs(y_lasso-Y_test_rev)**2)**0.5)
np.savetxt("revenue_lasso.csv", y_lasso, delimiter=",",fmt='%.1e')
print("LASSO is over")


#print("SVR rbf is over")
#y_lin_rev = svr_lin.fit(X_train_rev, Y_train_rev).predict(X_test_rev)
#print(np.mean((y_lin_rev-Y_test_rev)**2))
#print("SVR lin is over")
#y_poly = svr_poly.fit(X_train, Y_train).predict(X_test)
#print(y_poly)
#print("SVR poly is over")
"""
#==============================================================CLASSIFICATION==================================================================

#=================================================================RATING=======================================================================

# svm classification
clf = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0).fit(X_train_rate, Y_train_rate.astype('int'))
y_predicted = clf.predict(X_test_rate)

# performance
print("SVM")
print("Classification report")

print(metrics.classification_report(Y_test_rate.astype('int'), y_predicted))

#print("Confusion matrix")
#print(metrics.confusion_matrix(Y_test_rate.astype('int'), y_predicted))



#Bagging
model = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
model.fit(X_train_rate,Y_train_rate.astype('int'))
y_predicted = model.predict(X_test_rate)

# performance
print("Bagging")
print("Classification report")

print(metrics.classification_report(Y_test_rate.astype('int'), y_predicted))

#print("Confusion matrix")
#print(metrics.confusion_matrix(Y_test_rate.astype('int'), y_predicted))


#Adaboost
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train_rate,Y_train_rate.astype('int'))
y_predicted = model.predict(X_test_rate)

# performance
print("Adaboost")
print("Classification report")

print(metrics.classification_report(Y_test_rate.astype('int'), y_predicted))

#print("Confusion matrix")
#print(metrics.confusion_matrix(Y_test_rate.astype('int'), y_predicted))

#KNN
model = KNeighborsClassifier()
model.fit(X_train_rate,Y_train_rate.astype('int'))
y_predicted = model.predict(X_test_rate)

# performance
print("KNN")
print("Classification report")

print(metrics.classification_report(Y_test_rate.astype('int'), y_predicted))

#print("Confusion matrix")
#print(metrics.confusion_matrix(Y_test_rate.astype('int'), y_predicted))

#Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_rate, Y_train_rate.astype('int'))
y_predicted = model.predict(X_test_rate)

# performance
print("KNN")
print("Classification report")

print(metrics.classification_report(Y_test_rate.astype('int'), y_predicted))


