import math
import copy
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)   # protect against overflow
    g = 1.0/(1.0+np.exp(-z))

    return g


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    return cost / m


def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = sigmoid(np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:      # prevent resource exhaustion
            J_history.append(compute_cost(x, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w, b, J_history


features = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5],
                   [3, 0.5], [2, 2], [1, 2.5]])
targets = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.zeros_like(features[0])
b_tmp = 0.
alph = 0.1
iters = 10000

w_out, b_out, J_hist = gradient_descent(features, targets, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}",
      f"final cost: {J_hist[-1]}")
print(f"results: {sigmoid(np.dot(features, w_out) + b_out)}")