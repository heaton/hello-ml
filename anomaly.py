import numpy as np
import matplotlib.pyplot as plt
from utils_anomaly import *

def estimate_gaussian(X): 
  """
  Calculates mean and variance of all features 
  in the dataset

  Args:
    X (ndarray): (m, n) Data matrix

  Returns:
    mu (ndarray): (n,) Mean of all features
    var (ndarray): (n,) Variance of all features
  """

  mu = np.mean(X, axis=0)
  var = np.mean((X - mu) ** 2, axis=0)
  return mu, var

def select_threshold(y_val, p_val): 
  """
  Finds the best threshold to use for selecting outliers 
  based on the results from a validation set (p_val) 
  and the ground truth (y_val)

  Args:
    y_val (ndarray): Ground truth on validation set
    p_val (ndarray): Results on validation set

  Returns:
    epsilon (float): Threshold chosen 
    F1 (float):      F1 score by choosing epsilon as threshold
  """ 

  best_epsilon = 0
  best_F1 = 0
  F1 = 0

  step_size = (max(p_val) - min(p_val)) / 1000

  for epsilon in np.arange(min(p_val), max(p_val), step_size):
    predictions = p_val < epsilon

    tp = np.sum(y_val)
    fp = np.sum(predictions & (y_val == False))
    fn = np.sum((predictions == False) & y_val)

    prec = tp/(tp + fp)
    rec = tp/(tp + fn)

    F1 = 2 * prec * rec / (prec + rec)

    if F1 > best_F1:
        best_F1 = F1
        best_epsilon = epsilon

  return best_epsilon, best_F1

X_train, X_val, y_val = load_data()

plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 
plt.title("The first dataset")
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()

mu, var = estimate_gaussian(X_train)              

print("Mean of each feature:", mu)
print("Variance of each feature:", var)

p = multivariate_gaussian(X_train, mu, var)

visualize_fit(X_train, mu, var)
plt.show()

p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)

outliers = p < epsilon

visualize_fit(X_train, mu, var)

plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)
plt.show()

X_train_high, X_val_high, y_val_high = load_data_multi()

print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)

mu_high, var_high = estimate_gaussian(X_train_high)
# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))
