import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    temp = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y.flatten() == group, :]
        temp.append(Xg.mean(0))
        
    means=np.transpose(np.asarray(temp))
    global_mean=np.mean(X,axis=0)
    
    corrected = []
    for group in classes:
        Xg = X[y.flatten() == group, :]
        corrected.append(Xg - global_mean)
    
    covmatrix = []
    for group in classes:
        covmatrix.append((np.dot(np.transpose(corrected[int(group)-1]),corrected[int(group)-1]))/(corrected[int(group)-1].size/2))
    
    covmat=np.zeros((covmatrix[0].shape))
    for group in classes:
        multiplier=(corrected[int(group)-1].size/2.0)/(X.size/2.0)
        covmat=covmat+((covmatrix[int(group)-1])*multiplier)

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    temp = []
    classes = np.unique(y)
    for group in classes:
        Xg = X[y.flatten() == group, :]
        temp.append(Xg.mean(0))
        
    means=np.transpose(np.asarray(temp))
    global_mean=np.mean(X,axis=0)
    
    corrected = []
    for group in classes:
        Xg = X[y.flatten() == group, :]
        corrected.append(Xg - global_mean)
    
    covmats = []
    for group in classes:
        Xg = X[y.flatten() == group, :]
        Xg = Xg - global_mean
        covmats.append(np.cov(Xg, rowvar =0))   

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    covmat_deter=np.linalg.det(covmat);
    inverse_covmat=np.linalg.inv(covmat);
   
    pdf= np.zeros((Xtest.shape[0],means.shape[1]));
    for i in range(means.shape[1]):
        pdf[:,i] = np.exp(-0.5*np.sum((Xtest - means[:,i])*np.dot(inverse_covmat, (Xtest - means[:,i]).T).T,1))/(np.sqrt(np.pi*2)*(np.power(covmat_deter,2)));
    acc = 100*np.mean((np.argmax(pdf,1)+1) == ytest.reshape(ytest.size));
  #  ypred = 
    return acc #,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    pdf= np.zeros((Xtest.shape[0],means.shape[1]))
    for eachValue in range (means.shape[1]):
        inverse_covmat = np.linalg.inv(covmats[eachValue]);
        covmat_deter = np.linalg.det(covmats[eachValue]);
        pdf[:,eachValue] = np.exp(-0.5*np.sum((Xtest - means[:,eachValue])*np.dot(inverse_covmat, (Xtest - means[:,eachValue]).T).T,1))/(np.sqrt(np.pi*2)*(np.power(covmat_deter,2)));
    acc = 100*np.mean((np.argmax(pdf,1)+1) == ytest.reshape(ytest.size)); 
   # ypred = 
    return acc #,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD  
   # X_T = np.transpose(X)
   # inv_part = np.linalg.inv(np.dot(X_T,X))
   # w = np.dot(inv_part,np.dot(X_T,y)) 
    X_T = np.transpose(X)
    w =  np.dot(np.linalg.inv(np.dot(X_T,X)),np.dot(X_T,y))                                                 
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD 
   # X_T = np.transpose(X)
   # identity = np.eye(X.shape[1])
   # h_1 = np.dot(X_T,X) 
   # h_2 = lambd * identity 
    #h_3 = np.sum(h_1,h_2)
    #inverse = np.linalg.inv(h_3)  
    Identity = np.eye(X.shape[1]) 
    X_T = np.transpose(X)
    N = X.shape[0]
    w =  np.dot(np.linalg.inv(np.add(np.dot(X_T,X),(N*lambd*Identity))),np.dot(X_T,y))               
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
   # w_T = np.transpose(w)
   # k_1 = np.dot(Xtest,w)
   # k_2 = np.subtract(ytest,k_1)
   # mse = (np.sum((k_2)*(k_2)))/ Xtest.shape[0]
    term = np.dot(Xtest,w)
    N = Xtest.shape[0]
    mse=(np.sum((np.subtract(ytest,term)*np.subtract(ytest,term))))/N
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD  
    #w_T = np.transpose(w)
   # term = np.subtract(y,np.dot(X,w))
    
   # r_2 = 0.5 * r_1
   # r_3 = np.dot(w_T,w)
   # r_4 = 0.5 * lambd * r_3
   # error = r_2 + r_4
    w_T=np.array([w]).T
    r_1 = np.dot(X,w_T)
    y_T = np.transpose(y)
    error=((np.dot(np.transpose(np.subtract(y,r_1)),np.subtract(y,r_1)))/(2.0*X.shape[0])) + ((np.dot(lambd,np.dot(np.transpose(w_T),w_T)))/2) 
    error_grad = ((-np.dot(y_T,X)+np.dot(w_T,np.dot(np.transpose(X),X)))/X.shape[0]) + np.dot(lambd,w_T)
                                               
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    N = x.shape[0]
    Xp = np.ones(x.shape[0],p+1)
    for i in range (1,p+1):
        Xp[:,i] = x**i
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries

x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0], x2.shape[0])),alpha=0.3) 
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0], x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept for train '+str(w))
print('MSE with intercept for train '+str(w_i))

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    mses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.show()
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.legend(('No Regularization','Regularization'))
plt.show()
