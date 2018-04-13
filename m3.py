import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from numpy.linalg import inv
import pandas as pd
import math

def sigmoid(H):
	H_sig = H*(-1.0)
	H_sig = np.exp(H_sig)
	H_sig = 1.0/(1 + H_sig)
	return H_sig

def loglikelihood(Theta, X, Y):
	H = np.dot(X,Theta)
	H_sig = sigmoid(H)
	first = np.log(H_sig)
	second = np.log(1-H_sig)
	first = np.dot(Y.transpose(),first)
	second = np.dot((1-Y).transpose(), second)
	return (first + second)

def gradient(Theta, X, Y):
	diff = np.subtract(Y, sigmoid(np.dot(X,Theta)))
	gradient = np.dot(X.transpose(),diff)
	return gradient

def hessian(Theta, X):
	H_sig = sigmoid(np.dot(X,Theta))
	prod = np.multiply(H_sig, (1-H_sig))
	D = np.zeros((100,100))
	for i in range(0,100):
		D[i,i] = prod[i,0]
	temp = np.dot(X.transpose(), D)
	return np.dot(temp,X)

def newton_method(Theta, X, Y):
	i = 0
	print('Initial Loglikelihood - ', loglikelihood(Theta, X, Y))
	prev = loglikelihood(Theta, X, Y)
	while(True):
		update = np.dot(inv(hessian(Theta,X)), gradient(Theta, X, Y))
		Theta = np.add(Theta, update)
		i = i + 1
		ll_theta = loglikelihood(Theta, X, Y)
		## Terminating condition
		if(abs(ll_theta-prev) < 0.000000001):
			break
		print('Iteration - ',i,' : Loglikelihood - ', loglikelihood(Theta, X, Y))
		prev = ll_theta
	return Theta

def main():

	## Reading data
	file = open('logisticX.csv')
	X = np.loadtxt(file, delimiter= ',', skiprows=0)
	Y = np.genfromtxt('logisticY.csv', delimiter='\n').reshape(100,1)
	
	## Data Preprocessing
	for i in range(0,2):
		mean = np.mean(X[:,i]);
		var = np.var(X[:,i]);
		X[:,i] = (X[:,i]-mean)/math.sqrt(var);

	## Adding intercept term
	X_temp = np.zeros((X.shape[0],1))
	X_temp[:,0] = 1.0
	X = np.hstack((X_temp, X))

	## Initialize Theta
	Theta = np.zeros([3, 1], dtype = float)
	Theta[0,0] = 0.0
	Theta[1,0] = 0.0
	Theta[2,0] = 0.0

	## Run Newton's Method
	Theta = newton_method(Theta, X, Y)
	print(Theta)

	## Plot the given data
	df = pd.DataFrame(dict(A=X[:,1].transpose(),
                       B=X[:,2].transpose(),
                       C=Y.transpose().flatten()))

	plt.scatter(df.A[df.C == 1], df.B[df.C == 1], c='red', marker='o', label='Y = 1')
	plt.scatter(df.A[df.C == 0], df.B[df.C == 0], c='aqua', marker='D', label='Y = 0')
	ex = X.transpose()[1,:]
	ex_temp = np.ones((100))
	ex = np.vstack((ex_temp, ex))
	ey = np.dot(Theta[0:2,:].transpose(), ex)
	ey = ((-1.0)*ey)/Theta[2,0]
	print(ey.shape)
	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.title('Logistic Regression')
	plt.plot(ex[1,:],ey.transpose(), label='Decision Boundary')
	legend = plt.legend(loc='lower right', fontsize='small')
	plt.show()
	

if __name__ == "__main__":
    main()