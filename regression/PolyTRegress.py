import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Function used for feature manipulation/combination.
def manipulateFeatures(data):
	for i in range(len(data)):
		# Square MYCT
		data[i][0] = data[i][0] * data[i][0]
		# Combine MMIN and MMAX
		data[i][2] = data[i][2] + data[i][1]
		# Combine CHMIN and CHMAX
		data[i][5] = data[i][5] + data[i][4]
	# Delete MMIN
	data = np.delete(data, np.s_[1:2], axis=1)
	# Delete CHMAX
	data = np.delete(data, np.s_[3:4], axis=1)

	return data

def main():
	# Load Data Sets.
	data_x_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
	data_y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]
	X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)

	# Split training data for validation.
	X_train, X_val, Y_train, Y_val = train_test_split(data_x_train, data_y_train, test_size = 0.2, random_state = 0)

	# Feature manipulation.
	X_train = manipulateFeatures(X_train)
	X_val = manipulateFeatures(X_val)
	X_test = manipulateFeatures(X_test)

	# Scale X data.
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_val = scaler.transform(X_val)
	X_test = scaler.transform(X_test)

	# Scale Y data.
	scalery = preprocessing.StandardScaler().fit(Y_train)
	Y_train = scalery.transform(Y_train)
	Y_val = scalery.transform(Y_val)

	# Test to find best degree for polynomial.
	bestRMSE = 1e10
	bestDegree = 0
	degrees = np.arange(1, 10)
	for degreeIndex in range(1,10):
		d_clf = linear_model.LinearRegression()
		d_poly = preprocessing.PolynomialFeatures(degree=degreeIndex)
		X_ = d_poly.fit_transform(X_train)
		X_test_ = d_poly.fit_transform(X_val)

		d_clf.fit(X_, Y_train)

		y_pred = d_clf.predict(X_test_)

		rmse = np.sqrt(mean_squared_error(y_pred, Y_val))
		print("At degree " + str(degreeIndex) + " RMSE = " + str(rmse))

		if (rmse < bestRMSE):
			bestRMSE = rmse
			bestDegree = degreeIndex
		# Degree of 3 seems to give the best results on Kaggle.
		bestDegree = 3

	print("Best RMSE of " + str(bestRMSE))
	print("Using Degree " + str(bestDegree))

	# Train model using training data.
	clf = linear_model.LinearRegression()
	poly = preprocessing.PolynomialFeatures(degree=bestDegree)
	X_ = poly.fit_transform(X_train)
	X_test_poly = poly.fit_transform(X_test)
	clf.fit(X_, Y_train)
	y_pred = clf.predict(X_test_poly)

	# Make prediction.
	y_pred = scalery.inverse_transform(y_pred)

	print(y_pred)

	test_header = "Id,PRP"
	n_points = X_test_poly.shape[0]
	y_pred_pp = np.ones((n_points, 2))
	y_pred_pp[:, 0] = range(n_points)
	y_pred_pp[:, 1] = y_pred
	np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",", header=test_header, comments="")
main()