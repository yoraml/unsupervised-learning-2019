import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from data_inference.dimensions.dimensional_reducer import DimensionalReducer


def plot_pca_readmitted_scattering(data_set, dimensions):
    """
    plotting pca scatter against readmitted labels
    :param data_set: diabetic data set path
    :param dimensions: dimensions number you wish to reduce to
    :return: None
    """
    diabetic_data = pd.read_csv(data_set)
    labels = diabetic_data['readmitted']
    diabetic_data = DimensionalReducer(diabetic_data).reduce_dimension(dimensions)

    plt.scatter(diabetic_data[:, 0], diabetic_data[:, 1], diabetic_data[:, 2], c=labels)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(diabetic_data[:, 0], diabetic_data[:, 1], diabetic_data[:, 2], c=labels)
    plt.show()


if __name__ == '__main__':
    # found by the script "optimal_clusters_num_provider.py" that clusters size 3
    # is optimal for modified data with medicines
    plot_pca_readmitted_scattering("../../data/processed_data/latest_for_clustering_version2.csv", dimensions=3)
