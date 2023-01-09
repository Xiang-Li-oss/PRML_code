import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        content = f.readlines()
        for i in content:
            i = i[:-1].split(" ")
            data.append([float(i[0]), float(i[1])])
    data = np.array(data)
    return data


class SpectralClustering:
    def __init__(self, data, n_cluster=2):
        self.n_cluster = n_cluster
        self.data = data
        self.n_samples = data.shape[0]

    def fit(self, sigma=1, k=10):
        W, L_sym = self.generate_graph(sigma, k)

        e_value, e_vector = np.linalg.eig(L_sym)
        idxs = np.argsort(e_value)[:self.n_cluster]
        U = e_vector[:, idxs]

        for val in U:
            val = val / (val ** 2) ** 0.5

        kmeans = KMeans(n_clusters=2)
        label = kmeans.fit_predict(U)

        return label

    def generate_graph(self, sigma=1, k=10):
        W = np.zeros([self.n_samples, self.n_samples])

        for i in range(self.n_samples):
            weights = []
            for j in range(self.n_samples):
                w_ij = np.exp(-((data[i] - data[j]) ** 2).sum() / (2 * sigma ** 2))
                weights.append(w_ij)

            weights = np.array(weights)
            weights[i] = 0
            idxs = np.argsort(weights)[-k:]

            for idx in idxs:
                W[i, idx] = W[idx, i] = weights[idx]

        W = (W + W.T) / 2

        D = np.diag(np.sum(W, axis=-1))

        sqrt_D = np.diag(np.sum(W, axis=-1) ** -0.5)

        L = D - W
        L_sym = sqrt_D @ L @ sqrt_D

        return W, L_sym


data = load_data("./data.txt")
data = np.array(data)

model = SpectralClustering(data)

sigmas = (0.01, 0.1, 1, 10)
ks = (1, 5, 10, 20, 40, 50)
accs = []

for sigma in sigmas:
# for sigma in (10,):
#     for k in (10,):
    for k in (10,):
        pred_labels = model.fit()

        idx0 = np.where(pred_labels == pred_labels[0])
        idx1 = np.where(pred_labels == 1 - pred_labels[0])

        n_error = sum(idx0[0] >= 100) + sum(idx1[0] < 100)

        accs.append(1 - n_error / 100)

plt.plot(sigmas, accs)
plt.title('acc-sigma (k=10)')
plt.show()
