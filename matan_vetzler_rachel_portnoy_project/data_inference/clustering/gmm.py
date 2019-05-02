import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture.gaussian_mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D

from data_inference.dimensions.dimensional_reducer import DimensionalReducer

GMM_NUM_ITERATIONS = 20


def apply_GMM_clustering(data_set, dimensions, clusters):
    diabetic_data = pd.read_csv(data_set)
    diabetic_data = DimensionalReducer(diabetic_data).reduce_dimension(dimensions)
    model = GaussianMixture(n_components=clusters,
                             n_init=GMM_NUM_ITERATIONS)
    labels = model.fit_predict(diabetic_data)

    print(diabetic_data)
    print(diabetic_data[0])
    plt.scatter(diabetic_data[:, 0], diabetic_data[:, 1], diabetic_data[:, 2], c=labels)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(diabetic_data[:, 0], diabetic_data[:, 1], diabetic_data[:, 2], c=labels)
    plt.show()


if __name__ == '__main__':
    # found by the script "optimal_clusters_num_provider.py" that clusters size 3
    # is optimal for modified data with medicines
    apply_GMM_clustering("../../data/processed_data/latest_for_clustering_version2.csv", dimensions=3, clusters=3)
