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



genre = ['Action','Adventure','Fantasy','Sci-Fi','Thriller','Romance','Animation','Comedy','Family','Fantasy','Musical','Mystery','Western','Drama','Sport','Crime','Horror','History','War','Biography']

total = 586.0


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


with open('train.csv', 'r') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		col0.append(row[0])
		col1.append(row[1])

		cf9.append(row[2])
		cf10.append(row[3])
		cf11.append(row[4])
		cf12.append(row[5])

		col6.append(row[6])

		cf13.append(row[7])
		cf14.append(row[8])

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
d_genre = {}
count = 0
for ele in col9:
	for gen in genre:
		if gen in ele:
			if gen in d_genre:
				#print(genre)
				d_genre[gen] += 1
			else: 
				d_genre[gen] = 1
			count += 1

#print(d_genre)
#print(count)
genre_col_final = []
for ele in col9:
	val=0
	for gen in genre:
		if gen in ele:
			val += float(d_genre[gen])/float(count)
	genre_col_final.append(val)
			
#print(genre_col_final)


col0_dict = Counter(col0)
col1_dict = Counter(col1)
col6_dict = Counter(col6)
#col9_dict = Counter(col9)
col10_dict = Counter(col10)
col13_dict = Counter(col13)
col16_dict = Counter(col16)
col17_dict = Counter(col17)
col18_dict = Counter(col18)

for key in col0_dict:
	col0_dict[key] = float(col0_dict[key])/total

for key in col1_dict:
	col1_dict[key] = float(col1_dict[key])/total

for key in col6_dict:
	col6_dict[key] = float(col6_dict[key])/total

for key in col10_dict:
	col10_dict[key] = float(col10_dict[key])/total

for key in col13_dict:
	col13_dict[key] = float(col13_dict[key])/total

for key in col16_dict:
	col16_dict[key] = float(col16_dict[key])/total

for key in col17_dict:
	col17_dict[key] = float(col17_dict[key])/total

for key in col18_dict:
	col18_dict[key] = float(col18_dict[key])/total

cf0 = []
cf1 = []
cf2 = []
cf3 = genre_col_final
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
cf14 = np.array(cf14).astype(np.float)
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

X = np.column_stack((cf0,cf1,cf2,cf3,cf4,cf5,cf6,cf7,cf8,cf9,cf10,cf11,cf12,cf13,cf14,cf15,cf16,cf17,cf18,cf19,cf20,cf21,cf23,cf24))
Y = cf22

#print(X)
np.resize(X,(586,24))
np.resize(Y,(586,1))
#print(X.shape)
#print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

print(np.mean((regr.predict(X_test)-Y_test)**2))

"""
X_test = np.resize(X_test,(194,24))
Y_test = np.resize(Y_test,(194,1))
# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.plot(X_test, regr.predict(X_test), color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
print("LR is over")
"""


svr_rbf = SVR(kernel='rbf', C=0.1, gamma=0.2)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X_train, Y_train).predict(X_test)
#print(y_rbf)
print(np.mean((y_rbf-Y_test)**2))
#print("SVR rbf is over")
#y_lin = svr_lin.fit(X_train, Y_train).predict(X_test)
#print(y_lin)
#print("SVR lin is over")
#y_poly = svr_poly.fit(X_train, Y_train).predict(X_test)
#print(y_poly)
#print("SVR poly is over")

