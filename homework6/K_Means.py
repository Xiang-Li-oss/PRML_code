from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

cov = np.array([[1, 0],
                [0, 1]])
n = 200

mu1 = np.array([1, -1])
x1 = np.random.multivariate_normal(mu1, cov, n, 'raise')

mu2 = np.array([5.5, -4.5])
x2 = np.random.multivariate_normal(mu2, cov, n, 'raise')

mu3 = np.array([1, 4])
x3 = np.random.multivariate_normal(mu3, cov, n, 'raise')

mu4 = np.array([6, 4.5])
x4 = np.random.multivariate_normal(mu4, cov, n, 'raise')

mu5 = np.array([9, 0])
x5 = np.random.multivariate_normal(mu5, cov, n, 'raise')


class K_Means():
    def __init__(self, data, n_cluster=5):
        self.n_cluster = n_cluster

        self.data = data

        self.shape = shape = data.shape
        self.N = shape[0] * shape[1]

        self.n_labels = shape[0]

        self.cluster_means = None

    def fit(self, init_type=1):
        if init_type == 1:
            mu = np.mean(self.data.reshape([-1, 2]), axis=0)
            self.cluster_means = np.random.multivariate_normal(mu, cov, self.n_cluster, 'raise')
        else:
            self.cluster_means = np.random.random([5, 2])

        print(self.cluster_means)
        loss_history = []

        while True:
            points_idx_to_cluster, cluster_to_points, loss = self.re_assign()

            loss_history.append(loss / self.N)

            print(f'loss: {loss_history[-1]}')

            if len(loss_history) >= 2 and loss_history[-2] - loss_history[-1] < 1e-8:
                print('stop')
                break

            # update
            for key, item in cluster_to_points.items():
                self.cluster_means[key] = sum(item) / len(item)

        cluster_idx_to_label = self.map_cluster_to_label()

        true_labels = np.array([[i] * n for i in range(self.n_labels)]).reshape(-1)

        pred_labels = np.array([cluster_idx_to_label[point_idx_to_cluster]
                                for point_idx_to_cluster in points_idx_to_cluster])

        acc = accuracy_score(true_labels, pred_labels)

        return acc, pred_labels, self.cluster_means, loss_history

    def re_assign(self):
        cluster_to_points = defaultdict(list)
        points_idx_to_cluster = []
        loss = 0
        for label in range(self.n_labels):
            for point in self.data[label]:
                min_dis = np.inf
                cluster_belonged = None
                for cluster_idx, cluster_mean in enumerate(self.cluster_means):
                    dis = ((cluster_mean - point) ** 2).sum()
                    if dis < min_dis:
                        min_dis = dis
                        cluster_belonged = cluster_idx
                loss += min_dis
                cluster_to_points[cluster_belonged].append(point)
                points_idx_to_cluster.append(cluster_belonged)

        return points_idx_to_cluster, cluster_to_points, loss

    def map_cluster_to_label(self):
        idx_to_label = defaultdict(int)
        for idx, cluster_mean in enumerate(self.cluster_means):
            min_dis = np.inf
            label = None
            for i, mu in enumerate([mu1, mu2, mu3, mu4, mu5]):
                dis = (((cluster_mean - mu) ** 2) ** 0.5).sum()
                if dis < min_dis:
                    min_dis = dis
                    label = i
            idx_to_label[idx] = label

        return idx_to_label


def plot(cluster_means, title):
    plt.scatter(x1[:, 0], x1[:, 1], s=5)
    plt.scatter(x2[:, 0], x2[:, 1], s=5)
    plt.scatter(x3[:, 0], x3[:, 1], s=5)
    plt.scatter(x4[:, 0], x4[:, 1], s=5)
    plt.scatter(x5[:, 0], x5[:, 1], s=5)
    for i in range(5):
        plt.scatter(cluster_means[i, 0], cluster_means[i, 1], marker='+')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (6, 5)

    data = np.array([x1, x2, x3, x4, x5])
    model = K_Means(data)

    acc, pred_labels, cluster_means, loss_history = model.fit(init_type=1)
    plot(cluster_means, f'acc: {acc}, loss: {loss_history[-1]:.2f}')

    acc, pred_labels, cluster_means, loss_history = model.fit(init_type=2)
    plot(cluster_means, f'acc: {acc}, loss: {loss_history[-1]:.2f}')

