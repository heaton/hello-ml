import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

np.set_printoptions(precision=2)


def split_data(x, y):
    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
    x_train, x_, y_train, y_ = train_test_split(
        x, y, test_size=0.40, random_state=1)
    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_, y_, test_size=0.50, random_state=1)
    return x_train, y_train, x_cv, y_cv, x_test, y_test


def mse(y_train, yhat):
    total_squared_error = 0

    for i in range(len(yhat)):
        squared_error_i = (yhat[i] - y_train[i])**2
        total_squared_error += squared_error_i

    return total_squared_error / (2*len(yhat))


def plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title):
    plt.scatter(x_train, y_train, marker='x', c='r', label='training')
    plt.scatter(x_cv, y_cv, marker='o', c='b', label='cross validation')
    plt.scatter(x_test, y_test, marker='^', c='g', label='test')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def scale(x_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    return scaler, X_train_scaled


def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model


def sl_mse(model, x, y):
    yhat = model.predict(x)
    return mean_squared_error(y, yhat) / 2


def plot_train_cv_mses(degrees, train_mses, cv_mses, title):
    plt.plot(degrees, train_mses, marker='o', c='r', label='training MSEs')
    plt.plot(degrees, cv_mses, marker='o', c='b', label='CV MSEs')
    plt.title(title)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()


print("======= Linear Regression ========")
data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')

x = data[:, 0]
y = data[:, 1]

x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

print(f"the shape of the inputs x is: {x.shape}")
print(f"the shape of the targets y is: {y.shape}")

x_train, y_train, x_cv, y_cv, x_test, y_test = split_data(x, y)

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test,
                   y_test, title="input vs. target")

scaler_linear, X_train_scaled = scale(x_train)
linear_model = linear_regression(X_train_scaled, y_train)

# Evaluate the model
# mean squared error (MSE) for training set
yhat = linear_model.predict(X_train_scaled)
print(
    f"training MSE (sklearn function): {mean_squared_error(y_train, yhat) / 2}")
print(f"training MSE (implementation): {mse(y_train, yhat).squeeze()}")

# MSE for CV set
X_cv_scaled = scaler_linear.transform(x_cv)
yhat = linear_model.predict(X_cv_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")

print("====== Add polynomial features ======")


def train_poly_model(degree, x_train, y_train):
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    scaler, X_train_mapped_scaled = scale(X_train_mapped)
    model = linear_regression(X_train_mapped_scaled, y_train)
    train_mse = sl_mse(model, X_train_mapped_scaled, y_train)

    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
    cv_mse = sl_mse(model, X_cv_mapped_scaled, y_cv)
    return train_mse, cv_mse, model, scaler


train_mse, cv_mse, _, _ = train_poly_model(2, x_train, y_train)

print(f"Training MSE: {train_mse}")
print(f"Cross validation MSE: {cv_mse}")

print("======== More additional features =========")
train_mses = []
cv_mses = []
models = []
scalers = []

for degree in range(1, 11):
    train_mse, cv_mse, model, scaler = train_poly_model(
        degree, x_train, y_train)
    train_mses.append(train_mse)
    cv_mses.append(cv_mse)
    models.append(model)
    scalers.append(scaler)


plot_train_cv_mses(range(1, 11), train_mses, cv_mses,
                   title="degree of polynomial vs. train and CV MSEs")

degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")

poly = PolynomialFeatures(degree, include_bias=False)
X_test_mapped = poly.fit_transform(x_test)
X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)
test_mse = sl_mse(models[degree-1], X_test_mapped_scaled, y_test)

print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")

print("======== Neural Network ========")


def build_models():
    tf.random.set_seed(20)
    model_1 = Sequential(
        [
            Dense(25, activation='relu'),
            Dense(15, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_1'
    )
    model_2 = Sequential(
        [
            Dense(20, activation='relu'),
            Dense(12, activation='relu'),
            Dense(12, activation='relu'),
            Dense(20, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_2'
    )
    model_3 = Sequential(
        [
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(4, activation='relu'),
            Dense(12, activation='relu'),
            Dense(1, activation='linear')
        ],
        name='model_3'
    )
    return [model_1, model_2, model_3]


poly = PolynomialFeatures(degree=1, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)

scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)

nn_train_mses = []
nn_cv_mses = []

nn_models = build_models()

for model in nn_models:
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    )

    print(f"Training {model.name}...")

    model.fit(
        X_train_mapped_scaled, y_train,
        epochs=300,
        verbose=0
    )

    print("Done!\n")

    train_mse = sl_mse(model, X_train_mapped_scaled, y_train)
    nn_train_mses.append(train_mse)

    cv_mse = sl_mse(model, X_cv_mapped_scaled, y_cv)
    nn_cv_mses.append(cv_mse)


print("RESULTS:")
for model_num in range(len(nn_train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
        f"CV MSE: {nn_cv_mses[model_num]:.2f}"
    )

# Select the model with the lowest CV MSE
model_num = 3
test_mse = sl_mse(nn_models[model_num-1], X_test_mapped_scaled, y_test)

print(f"Selected Model: {model_num}")
print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")

print("======== Classification ========")


def misclassfied_error(model, x, y, threshold=0.5):
    yhat = model.predict(x)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= threshold, 1, 0)
    return np.mean(yhat != y)


data = np.loadtxt('./data/data_w3_ex2.csv', delimiter=',')

x_bc = data[:, :-1]
y_bc = data[:, -1]

y_bc = np.expand_dims(y_bc, axis=1)

print(f"the shape of the inputs x is: {x_bc.shape}")
print(f"the shape of the targets y is: {y_bc.shape}")


def plot_bc_dataset(x, y, title):
    for i in range(len(y)):
        marker = 'x' if y[i] == 1 else 'o'
        c = 'r' if y[i] == 1 else 'b'
        plt.scatter(x[i, 0], x[i, 1], marker=marker, c=c)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    y_0 = mlines.Line2D([], [], color='r', marker='x',
                        markersize=12, linestyle='None', label='y=1')
    y_1 = mlines.Line2D([], [], color='b', marker='o',
                        markersize=12, linestyle='None', label='y=0')
    plt.title(title)
    plt.legend(handles=[y_0, y_1])
    plt.show()


plot_bc_dataset(x_bc, y_bc, "x1 vs. x2")

x_bc_train, y_bc_train, x_bc_cv, y_bc_cv, x_bc_test, y_bc_test = split_data(
    x_bc, y_bc)

print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")

scaler_linear = StandardScaler()
x_bc_train_scaled = scaler_linear.fit_transform(x_bc_train)
x_bc_cv_scaled = scaler_linear.transform(x_bc_cv)
x_bc_test_scaled = scaler_linear.transform(x_bc_test)

nn_train_error = []
nn_cv_error = []

models_bc = build_models()

for model in models_bc:
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    print(f"Training {model.name}...")

    # Train the model
    model.fit(
        x_bc_train_scaled, y_bc_train,
        epochs=200,
        verbose=0
    )

    print("Done!\n")

    train_error = misclassfied_error(model, x_bc_train_scaled, y_bc_train)
    nn_train_error.append(train_error)

    cv_error = misclassfied_error(model, x_bc_cv_scaled, y_bc_cv)
    nn_cv_error.append(cv_error)

for model_num in range(len(nn_train_error)):
    print(
        f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
        f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
    )

# Select the lowest CV error
model_num = 3

nn_test_error = misclassfied_error(
    models_bc[model_num-1], x_bc_test_scaled, y_bc_test)

print(f"Selected Model: {model_num}")
print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
print(f"Test Set Classification Error: {nn_test_error:.4f}")
