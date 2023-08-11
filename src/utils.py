import numpy as np
import random


def create_support_set(x_train, y_train, classes, n):
    x_support = []
    y_support = []
    for c in classes:
        indices = np.where(np.array(y_train) == c)
        for i in range(0, n):
            x_support.append(x_train[indices[0][i]])
            y_support.append(y_train[indices[0][i]])
    return np.array(x_support), np.array(y_support)


def create_couples(x_support, y_support, x_train, y_train):
    x_train_left = []
    x_train_right = []
    y_train_set = []
    for i in range(0, len(x_support)):
        for j in range(0, len(x_train)):
            x_train_left.append(x_support[i])
            x_train_right.append(x_train[j])
            if y_support[i] == y_train[j]:
                y_train_set.append(1)
            else:
                y_train_set.append(0)

    return np.array(x_train_left), np.array(x_train_right), np.array(y_train_set)


def generate_random_colors(num_colors):
    random_colors = []
    for _ in range(num_colors):
        random_color = "#{:02x}{:02x}{:02x}".format(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        random_colors.append(random_color)
    return random_colors
