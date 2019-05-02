import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


class DimensionalReducer:
    def __init__(self, diabetic_data):
        self.diabetic_data = diabetic_data

    def reduce_dimensions(self):
        self.display_histograms()
        self.display_correlation()

        return self.reduce_dimension()

    def display_histograms(self):
        fig, axes = plt.subplots(7, 2, figsize=(5, 5))
        readmitted0 = self.diabetic_data[self.diabetic_data['readmitted'] == 0]
        readmitted1 = self.diabetic_data[self.diabetic_data['readmitted'] == 1]
        readmitted2 = self.diabetic_data[self.diabetic_data['readmitted'] == 2]
        ax = axes.ravel()

        columns = self.diabetic_data.columns[2:3]
        for i, column in enumerate(columns):
            _, bins = np.histogram(self.diabetic_data[column], bins=25)
            ax[i].hist(readmitted0[column], bins=bins, color='r', alpha=.5)  # red color for malignant class
            ax[i].hist(readmitted1[column], bins=bins, color='g', alpha=0.3)  # alpha is
            ax[i].hist(readmitted2[column], bins=bins, color='b', alpha=0.2)  # alpha is
            ax[i].set_title(column, fontsize=9)
            ax[i].axes.get_xaxis().set_visible(
                False)  # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
            ax[i].set_yticks(())
        ax[0].legend(['readmitted: no', 'readmitted: >30', 'readmitted: <30'], loc='best', fontsize=8)
        plt.tight_layout()  # let's make good plots
        plt.show()

    def display_correlation(self):
        diabetic_df = pd.DataFrame(self.diabetic_data.data,
                                   columns=self.diabetic_data.feature_names)  # just convert the scikit learn data-set to pandas data-frame.
        plot_count = 1
        for col1 in self.diabetic_data.feature_names[10:13]:
            for col2 in self.diabetic_data.feature_names[10:13]:
                if col1 == col2:
                    continue
                plt.subplot(3, 2, plot_count)  # fisrt plot
                plt.plot(diabetic_df[col1], diabetic_df[col2], '*',
                         color='magenta', label='check', alpha=0.3)
                plt.xlabel(col1, fontsize=12)
                plt.ylabel(col2, fontsize=12)

                plot_count += 1

        plt.tight_layout()
        plt.show()

    def reduce_dimension(self, dimension):
        """
        reducing data dimensional size to 20 features based on prior research
        :return: modified data after PCA dimensional reduction
        """
        pca = PCA(n_components=dimension)
        X_pca = pca.fit_transform(self.diabetic_data)

        ex_variance = np.var(X_pca, axis=0)
        ex_variance_ratio = ex_variance / np.sum(ex_variance)

        # plt.plot(range(146), np.cumsum(ex_variance_ratio), 'r')
        # plt.scatter(range(146), np.cumsum(ex_variance_ratio), c=np.cumsum(ex_variance_ratio))
        # plt.ylabel('explained variance ratio')
        # plt.xlabel('no. principal components')
        # plt.show()
        # sns.heatmap(np.log(pca.inverse_transform(np.eye(self.diabetic_data.shape[1]))), cmap="hot", cbar=False)

        return X_pca
