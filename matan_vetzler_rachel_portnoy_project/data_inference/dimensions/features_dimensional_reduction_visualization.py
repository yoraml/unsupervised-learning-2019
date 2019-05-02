import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

diabetic_data = pd.read_csv("../data/processed_data/latest_for_clustering_version2.csv")
readmitted = diabetic_data['readmitted']
pca = PCA()
transformed = pca.fit_transform(diabetic_data)


def visualize_dimensional_reduction(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c=readmitted)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')


plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()


visualize_dimensional_reduction(transformed[:, 0:2], np.transpose(pca.components_[0:2, :]))
plt.show()
print(pca.explained_variance_ratio_)
print(abs(pca.components_))
