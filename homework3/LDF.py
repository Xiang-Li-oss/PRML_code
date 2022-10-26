import numpy as np
import pandas as pd
import numpy.linalg as linalg
import matplotlib.pyplot as plt


def get_y_on_line(x, a):
    y = (-a[2] - a[0] * x) / a[1]
    return y


def plot(features, labels, a):
    if a[2] != 0:
        x = np.array([min(features[:, 0]) - 1, max(features[:, 0] + 1)])
        y = get_y_on_line(x, a)
        plt.plot(x, y, linewidth='0.5', color='#000000')
    plt.scatter(features[:, 0], features[:, 1], c=labels)
    plt.show()


def load_data(path='./data.txt', chosen_labels=[1, 2, 3, 4]):
    df = pd.read_csv(path, delimiter=' ', header=None)
    data = df.values

    features, labels = data[:, :2], data[:, 2]
    labels = labels.astype(np.int32)

    idxs = []
    for chose_label in chosen_labels:
        idxs.extend(list(range((chose_label - 1) * 10, chose_label * 10)))

    return features[idxs], labels[idxs]


def batch_perceptron(chosen_labels=[1, 2]):
    features, labels = load_data(chosen_labels=chosen_labels)

    print(features)

    arg_features = np.concatenate(
        [features, np.ones([features.shape[0], 1])], axis=-1)
    arg_features[10:20] = -arg_features[10:20]

    # print(arg_features)

    a = np.zeros(arg_features[0].shape)

    cnt = 0
    while True:
        error_set = []
        g_ys = []
        for feature in arg_features:
            g_y = (a * feature).sum()
            g_ys.append(g_y)

            if g_y <= 0:
                error_set.append(feature)

        print(f'error set: {error_set}')

        if len(error_set) == 0:
            break

        cnt += 1
        a += sum(error_set)

    print(f'g_y: {g_ys}')
    print(f'converged after {cnt} steps')
    print(f'a = {a}')

    plot(features, labels, a)


def Ho_Kashyap(chosen_labels=[1, 3]):
    features, labels = load_data(chosen_labels=chosen_labels)

    n = features.shape[0]

    arg_features = np.concatenate(
        [features, np.ones([n, 1])], axis=-1)
    arg_features[10:20] = -arg_features[10:20]

    d = arg_features.shape[1]

    Y = np.mat(arg_features)
    a = np.mat(np.zeros([d, 1]))
    b = np.mat(np.full(n, 0.1).reshape(-1, 1))

    learning_rate = 0.1
    epsilon = 1e-4
    b_min = 1e-3
    k_max = 10000

    for k in range(1, k_max + 1):
        e = Y * a - b

        if k % 1000 == 0:
            print(f'step {k} error: {np.abs(e).sum()}')

        e_plus = 0.5 * (e + np.abs(e))
        b = b + 2 * learning_rate * e_plus

        Y_pseudo_inverse = linalg.pinv(Y)
        a = Y_pseudo_inverse * b

        if np.abs(e).max() <= b_min:
            print(f'converged after {k} steps')
            print(a.reshape(-1))
            print(b.reshape(-1))
            break
    
    if k == k_max:
        print('no solution find')
    
    plot(features, labels, np.array(a).reshape(-1))


def multi_class_MSE():
    features, labels = load_data()

    labels = np.eye(4)[labels - 1]

    n = features.shape[0]

    arg_features = np.concatenate(
        [features, np.ones([n, 1])], axis=-1)

    train_idxs = []
    test_idxs = []
    for i in range(4):
        train_idxs.extend(list(range(i * 10, i * 10 + 8)))
        test_idxs.extend(list(range(i * 10 + 8, (i + 1) * 10)))

    train_features, train_labels = np.mat(
        features[train_idxs]), np.mat(labels[train_idxs])
    test_features, test_labels = np.mat(
        features[test_idxs]), labels[test_idxs]

    X, Y = train_features, train_labels

    W = linalg.pinv(X.transpose() * X) * X.transpose() * Y

    score = np.array(test_features * W)
    pred = np.argmax(score, axis=1)

    print(f'prediction score:\n {score}')
    print(f'prediction: {pred}')

    acc = sum(test_labels[range(8), pred] == 1) / 8

    print(f'accuracy: {acc}')


batch_perceptron([1, 2])
Ho_Kashyap()
Ho_Kashyap([2, 4])
multi_class_MSE()
