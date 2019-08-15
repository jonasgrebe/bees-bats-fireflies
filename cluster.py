import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from bees import BeesAlgorithm
from bat import BatAlgorithm
from firefly import FireflyAlgorithm


def clustering_loss(X, k):
    def data_clustering_loss(x):
        centers = np.split(x, k)
        dists = np.zeros((len(centers), len(X)))
        for i in range(len(centers)):
            dists[i] = np.sqrt(np.sum(np.square(X-centers[i]), axis=1))
        return np.sum(np.min(dists, axis=0))
    
    return data_clustering_loss
        
def get_responsibilities(X, centers):
    dists = np.zeros((len(centers), len(X)))
    for i in range(len(centers)):
        dists[i] = np.sqrt(np.sum(np.square(X-centers[i]), axis=1))
    return np.argmin(dists, axis=0)


k = 3
d = 4 * k
n = 100
range_min, range_max = -5.0, 5.0
T = 500

iris_data = load_iris()['data']
iris_data = PCA(4).fit_transform(iris_data)
iris_clustering_loss = clustering_loss(iris_data, k)

objective = 'min'
objective_fct = iris_clustering_loss


bat = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                   a=0.5, r=0.5, q_min=0.0, q_max=3.0)

bees = BeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     nb=50, ne=20, nrb=5, nre=10, shrink_factor=0.8, stgn_lim=5)

firefly = FireflyAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                           alpha=1.0, beta0=1.0, gamma=0.5)


solution, latency = bees.search(objective, objective_fct, T, visualize=True)
bees.plot_history()

solution_x, solution_y = solution
centers = np.split(solution_x, k)

respons = get_responsibilities(iris_data, centers)
data_pca = PCA(2).fit_transform(np.concatenate([iris_data, centers]))

iris_data_pca = data_pca[:-len(centers)]
centers_pca = data_pca[-len(centers):]

center_colors = ['blue', 'red', 'green', 'orange', 'yellow', 'black', 'gray', 'cyan']

for i in range(k):
    X_c = iris_data_pca[np.where(respons == i)]
    print(len(X_c))
    plt.scatter(X_c.T[0], X_c.T[1], c=center_colors[i])

for center in centers_pca:
    plt.scatter(center[0], center[1], c='black', marker='X')

