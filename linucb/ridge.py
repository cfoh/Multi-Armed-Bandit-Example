'''
This program illustrates the use of Ridge regression for linear regression.
It contains two implementations:
- A closed-form solution for Ridge regression.
- An online implementation of Ridge regression.
'''

import numpy as np

#########################################################
## Closed-form Ridge regression solver
##
## The coefficient vector β = (X^T X + λI)^-1 X^T y
## where:
## - X is the input matrix with shape (n_samples, n_features)
## - y is the target vector with shape (n_samples)
## - λ is the regularization parameter
## - I is the identity matrix of shape (n_features, n_features)
## - X^T is the transpose of X
## Note:
## - if the intercept β0 is not zero, the input X should
##   be augmented with a column of ones, 
##   i.e. X = [1, x1, x2, ...]
#########################################################

def ridge_regression(X, y, lambda_ridge):
    _, dim = X.shape
    coeffs = np.linalg.inv(X.T @ X + lambda_ridge * np.eye(dim)) @ X.T @ y
    return coeffs

#########################################################
## Online Ridge regression solver
## 
## Initialize X^T X = 0 and X^T y = 0
## for each sample xi, yi:
## - update X^T X += xi @ xi.T
## - update X^T y += yi * xi
## - compute coefficients β = (X^T X + λI)^-1 X^T y
## where:
## - xi is the i-th sample (column vector)
## - yi is the corresponding target value
## - λ is the regularization parameter
## - I is the identity matrix of shape (dim, dim)
## - β is the vector of coefficients (intercept and slopes)
## - X^T is the transpose of X
## Note: 
## - the intercept term is included in the system, i.e. if there are 2 features,
##   the dimension of the problem is 3 (2 features + intercept), and the coefficients
##   vector contains β0, β1 and β2 for the intercept, feature 1 and 2 respectively.
#########################################################

class OnlineRidgeRegression:
    
    def __init__(self, num_features, lambda_ridge):
        self.lambda_ridge = lambda_ridge
        self.dim = num_features+1                  # dimension of the problem
        self.XTX = np.zeros((self.dim, self.dim))  # initialize X^T X matrix
        self.XTy = np.zeros(self.dim)              # initialize X^T y vector
        self.coeffs = None                         # coefficients

    def update(self, xi, yi):
        xi = np.insert(xi,0,1).reshape(-1,1) # add intercept & reshape to column vector
        self.XTX += xi @ xi.T
        self.XTy += yi * xi.flatten()
        self.coeffs = np.linalg.inv(self.XTX + self.lambda_ridge*np.eye(self.dim)) @ self.XTy

    def predict(self, xi):
        xi = np.insert(xi,0,1).reshape(-1,1) # add intercept & reshape to column vector
        return self.coeffs @ xi.flatten()
    
    def get_coeffs(self):
        return self.coeffs

#########################################################
## generate synthetic data
#########################################################

np.random.seed(42)
n_samples = 100
n_features = 2
X = np.random.rand(n_samples, n_features) * 100  # input features, between 0 and 100
X_aug = np.hstack([np.ones((n_samples, 1)), X])  # add intercept column

true_coeffs = np.array([15, 2, -3])  # true coefficients β0, β1 and β2
y_true = X_aug @ true_coeffs                     # target values
y = y_true + np.random.normal(0, 10, n_samples)  # target values with noise

#########################################################
## run the closed-form solver and print results
#########################################################

ridge_coeffs = ridge_regression(X_aug, y, lambda_ridge=1.0)
i = 10 # test sample index
y_pred = ridge_coeffs @ X_aug[i]
print("Closed-Form Solution:")
print("  Final coefficients:", ridge_coeffs)
print("  True coefficients:", true_coeffs)
print(f"  Sample {i+1}: x = {X[i]}, y = {y[i]:.4f}, "
        f"y_true = {y_true[i]:.4f}, y_pred = {y_pred:.4f}")
print(f"{X[i]}")
print()

#########################################################
## run the online solver and print results
#########################################################

print("Online Solution:")
ridge_regression = OnlineRidgeRegression(num_features=n_features, lambda_ridge=1.0)

## make prediction for each incoming sample
for i in range(n_samples):
    xi = X[i]       # get a sample
    yi = y[i]       # get the corresponding target value
    ridge_regression.update(xi, yi)
    y_pred = ridge_regression.predict(xi)

    # print the outcome so far
    print(f"  Sample {i+1}: x = {xi}, y = {yi:.2f}, "
          f"y_true = {y_true[i]:.2f}, y_pred = {y_pred:.2f}")

print("  Final coefficients:", ridge_regression.get_coeffs())
print("  True coefficients:", true_coeffs)

