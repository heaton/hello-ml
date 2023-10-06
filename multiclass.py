import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import matplotlib as mpl


def my_softmax(z):
    ez = np.exp(z)
    sm = ez/np.sum(ez)
    return (sm)


print(my_softmax(np.array([1, 2, 3, 4])))

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(
    n_samples=2000, centers=centers, cluster_std=1.0, random_state=30)

model = Sequential(
    [
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(4, activation='softmax')
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train, y_train,
    epochs=10
)

p_nonpreferred = model.predict(X_train)
print(p_nonpreferred[:2])
print("largest value", np.max(p_nonpreferred),
      "smallest value", np.min(p_nonpreferred))


print("=========== Better version for softmax ===========")
preferred_model = Sequential(
    [
        Dense(25, activation='relu'),
        Dense(15, activation='relu'),
        Dense(4, activation='linear')
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),  # <-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train, y_train,
    epochs=10
)

p_preferred = preferred_model.predict(X_train)
sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred),
      "smallest value", np.min(sm_preferred))

for i in range(5):
    print(f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")

dkcolors = plt.cm.Paired((1, 3, 7, 9, 5, 11))
dkcolors_map = mpl.colors.ListedColormap(dkcolors)


def plt_mc_data(ax, X, y, classes,  class_labels=None, map=plt.cm.Paired,
                legend=False, size=50, m='o', equal_xy=False):
    """ Plot multiclass data. Note, if equal_xy is True, setting ylim on the plot may not work """
    for i in range(classes):
        idx = np.where(y == i)
        col = len(idx[0])*[i]
        label = class_labels[i] if class_labels else "c{}".format(i)
        ax.scatter(X[idx, 0], X[idx, 1],  marker=m,
                   color=map(col), vmin=0, vmax=map.N,
                   s=size, label=label)
    if legend:
        ax.legend()
    if equal_xy:
        ax.axis("equal")


def plot_cat_decision_boundary_mc(ax, X, predict, class_labels=None, legend=False, vector=True):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max()+0.5
    h = max(x_max-x_min, y_max-y_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]
    if vector:
        Z = predict(points)
    else:
        Z = np.zeros((len(points),))
        for i in range(len(points)):
            Z[i] = predict(points[i].reshape(1, 2))
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, linewidths=1)


def plt_cat_mc(X_train, y_train, model, classes):
    def model_predict(Xl): return np.argmax(model.predict(Xl), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    plt_mc_data(ax, X_train, y_train, classes, map=dkcolors_map, legend=True)
    plot_cat_decision_boundary_mc(ax, X_train, model_predict, vector=True)
    ax.set_title("model decision boundary")

    plt.xlabel(r'$x_0$')
    plt.ylabel(r"$x_1$")
    plt.show()


plt_cat_mc(X_train, y_train, model, 4)
