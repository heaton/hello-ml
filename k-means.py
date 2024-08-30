import numpy as np
import matplotlib.pyplot as plt
from utils import *

def find_closest_centroids(X, centroids):
  """
  Computes the centroid memberships for every example
  
  Args:
      X (ndarray): (m, n) Input values      
      centroids (ndarray): (K, n) centroids
  
  Returns:
      idx (array_like): (m,) closest centroids
  
  """

  idx = np.zeros(X.shape[0], dtype=int)

  for i in range(X.shape[0]):
      distance = [] 
      for j in range(centroids.shape[0]):
          norm_ij = np.linalg.norm(X[i] - centroids[j])
          distance.append(norm_ij)

      idx[i] = np.argmin(distance)
  return idx

def compute_centroids(X, idx, K):
  """
  Returns the new centroids by computing the means of the 
  data points assigned to each centroid.
  
  Args:
      X (ndarray):   (m, n) Data points
      idx (ndarray): (m,) Array containing index of closest centroid for each 
                      example in X. Concretely, idx[i] contains the index of 
                      the centroid closest to example i
      K (int):       number of centroids
  
  Returns:
      centroids (ndarray): (K, n) New centroids computed
  """
  
  m, n = X.shape
  centroids = np.zeros((K, n))
  
  for k in range(K):   
      points = X[idx == k]
      centroids[k] = np.mean(points, axis = 0)
  
  return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
  """
  Runs the K-Means algorithm on data matrix X, where each row of X
  is a single example
  """
  
  # Initialize values
  m, n = X.shape
  K = initial_centroids.shape[0]
  centroids = initial_centroids
  previous_centroids = centroids    
  idx = np.zeros(m)
  plt.figure(figsize=(8, 6))

  # Run K-Means
  for i in range(max_iters):
      print("K-Means iteration %d/%d" % (i, max_iters-1))
      idx = find_closest_centroids(X, centroids)
      if plot_progress:
          plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
          previous_centroids = centroids
          
      centroids = compute_centroids(X, idx, K)
  plt.show() 
  return centroids, idx

X = load_data()

print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)

initial_centroids = np.array([[3,3], [6,2], [8,5]])
idx = find_closest_centroids(X, initial_centroids)

print("First three elements in idx are:", idx[:3])

K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)

max_iters = 10
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)

# Random initialization
def kMeans_init_centroids(X, K):
  """
  This function initializes K centroids that are to be 
  used in K-Means on the dataset X
  
  Args:
      X (ndarray): Data points 
      K (int):     number of centroids/clusters
  
  Returns:
      centroids (ndarray): Initialized centroids
  """
  
  # Randomly reorder the indices of examples
  randidx = np.random.permutation(X.shape[0])
  centroids = X[randidx[:K]]
  
  return centroids

initial_centroids = kMeans_init_centroids(X, K)
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
