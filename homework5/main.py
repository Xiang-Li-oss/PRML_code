import numpy as np
import scipy.io as scio
from numpy import inf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


def knn(train_samples, train_labels, test_samples, test_labels):
    n_acc = 0
    for test_sample, test_label in zip(test_samples, test_labels):
        min_dist = inf
        min_idx = None
        for j, train_sample in enumerate(train_samples):
            dist = np.sqrt(((test_sample - train_sample) ** 2).sum())
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        if train_labels[min_idx] == test_label:
            n_acc += 1
    return n_acc / len(test_labels)


def pca(train_samples, test_samples, retain_dim=5):
    samples = np.concatenate([train_samples, test_samples], axis=0)

    cov = np.cov(samples, rowvar=False)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    # (m, d)
    P = eig_vecs.T[:retain_dim, :]

    train_samples = train_samples @ P.T
    test_samples = test_samples @ P.T

    return train_samples, test_samples


def lda(train_samples, train_labels, test_samples, test_labels, retain_dim=5):
    mu = np.concatenate([train_samples, test_samples], axis=0).mean(axis=0)

    labels = set(train_labels) | set(test_labels)

    label_to_samples = {
        label: np.concatenate((train_samples[train_labels == label], test_samples[test_labels == label]),
                              axis=0)
        for label in labels
    }

    within_class_scatter_mats = []

    between_class_scatter_mats = []

    for label, samples in label_to_samples.items():
        within_class_scatter_mats.append(np.cov(samples, rowvar=False))

        mu_j = samples.mean(axis=0)
        a = mu_j - mu
        between_class_scatter_mats.append(samples.shape[0] * np.outer(a, a))

    S_w = np.sum(within_class_scatter_mats, axis=0)

    S_b = np.sum(between_class_scatter_mats, axis=0)

    M = np.linalg.pinv(S_w) @ S_b

    eig_vals, eig_vecs = np.linalg.eig(M)

    # (m, d)
    P = eig_vecs.T[:retain_dim, :]

    train_samples = train_samples @ P.T
    test_samples = test_samples @ P.T

    return train_samples, test_samples


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def get_ORLData():
    data_file = './ORLData_25.mat'
    data = scio.loadmat(data_file)['ORLData'].T
    return data


def get_vehicle():
    data_file = './vehicle.mat'
    data = scio.loadmat(data_file)['UCI_entropy_data'][0][0][4].T
    return data


if __name__ == '__main__':
    data = get_vehicle()
    data = np.concatenate([standardization(data[:, :-1]), data[:, -1:]], axis=-1)
    train, test = train_test_split(data, test_size=0.2)
    train_samples, train_labels = train[:, :-1], train[:, -1]
    test_samples, test_labels = test[:, :-1], test[:, -1]
    train_labels, test_labels = train_labels.astype(int), test_labels.astype(int)

    ori_acc = knn(train_samples, train_labels, test_samples, test_labels)
    print(f'with out pca accuracy: {ori_acc}')

    dims = range(1, 100, 5)
    pca_accuracies = []
    lda_accuracies = []

    for dim in tqdm(dims):
        new_train, new_test = lda(train_samples, train_labels, test_samples, test_labels, dim)
        acc = knn(new_train, train_labels, new_test, test_labels)
        lda_accuracies.append(acc)

        new_train, new_test = pca(train_samples, test_samples, dim)
        acc = knn(new_train, train_labels, new_test, test_labels)
        pca_accuracies.append(acc)

    plt.plot(dims, pca_accuracies, label='pca-knn')
    plt.plot(dims, lda_accuracies, label='lda-knn')
    plt.plot(dims, [ori_acc] * len(dims), label='knn')

    plt.legend()
    plt.show()
