import gpflow
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler

def make_kaggle_file(y_pred):
    ofile  = open('./results/Kaggle_submision_guassian.csv', "wb")
    writer = csv.writer(ofile)
    writer.writerow(["Id","PRP"])
    for i,score in enumerate(y_pred):
        writer.writerow([i,int(round(score[0]))])

    ofile.close()

X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
y_train = (np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1])
Y_train = y_train.reshape(y_train.shape[0], -1)

X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)

# Feature Scaling (for some reason results in 0 values for all inputs.)
# sc_X = StandardScaler()
# X_train_scaled = sc_X.fit_transform(X_train)
# X_test_scaled = sc_X.transform(X_test)

# sc_Y = StandardScaler()
# Y_train_scaled = sc_Y.fit_transform(Y_train)

k = gpflow.kernels.Matern52(1, lengthscales=0.3)
m = gpflow.models.GPR(X_train_scaled,Y_train_scaled,kern=k)
m.likelihood.variance = 0.01
m.compile()

gpflow.train.ScipyOptimizer().minimize(m)
y_pred, var = m.predict_y(X_test)

def make_kaggle_file(y_pred):
    ofile  = open('./results/Kaggle_submision_guassian_w_fs.csv', "w", newline='')
    writer = csv.writer(ofile)
    writer.writerow(["Id","PRP"])
    for i,score in enumerate(y_pred):
        writer.writerow([i,int(round(score[0]))])

    ofile.close()

make_kaggle_file(y_pred)