import copy
import math
import numpy as np


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    return cost / (2 * m)


def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:      # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]} ",
                  f"dj_dw: {dj_dw}, dj_db: {dj_db} ",
                  f"w: {w}, b:{b}")

    return w, b, J_history


# trainning data
features = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
targets = np.array([460, 232, 178])

# initialize parameters
w_init = np.array([0.4, 18., -53., -26.])
b_init = 700

print("===========run gradient descent==============")
def run_gradient_descent(features, alpha, iterations):
    return gradient_descent(features, targets, w_init, b_init, alpha,
                                            iterations, compute_cost, compute_gradient)

w_final, b_final, J_hist = run_gradient_descent(features, alpha = 5.0e-7, iterations=10000)

print(f"(w,b) found by gradient descent: ({w_final}, {b_final:1.4f});",
      f"final cost: {J_hist[-1]}")


def zscore_normalize_features(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma

    return (x_norm, mu, sigma)

print("===========run gradient descent with nomalization==============")
x_norm, mu, sigma = zscore_normalize_features(features)
w_norm, b_norm, J_hist = run_gradient_descent(x_norm, alpha=1e-1, iterations=1000)

print(f"(w,b) found by gradient descent: ({w_norm},{b_norm:1.4f});",
      f"final cost: {J_hist[-1]}")
print(f"""compare:
      {targets}
      {np.dot(features, w_final) + b_final}
      {np.dot(x_norm, w_norm) + b_norm}
""")