import numpy as np


def d1_dot(x, w):
    n = x.shape[0]
    s = 0
    for i in range(n):
        s += x[i]*w[i]
    return s


x = np.array([1, 2])
w = np.array([3, 4])
print(f"np.dot({x}, {w}): ", np.dot(x, w))
print(f"d1_dot({x}, {w}): ", d1_dot(x, w))


def simple_matmul(X, W):
    r = X.shape[0]
    c = Y.shape[1]
    out = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            out[i, j] = d1_dot(X[i], Y[:, j])
    return out


X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
Y = np.array([[9, 8, 7, 6],
              [5, 4, 3, 2]])

print(f"np.matmul({X}, {Y}): ", np.matmul(X, Y))
print(f"simple_matmul({X}, {Y}): ", simple_matmul(X, Y))
