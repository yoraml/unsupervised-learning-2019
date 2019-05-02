import pandas as pd

from data_preprocessing.data_encoder import DataEncoder
from data_preprocessing.data_eraser import DataEraser


def main():
    """
    main function responsible for running data deletion and data encoding methods before
    using clustering methods and saving the processed data onto designated csv file
    :return: None
    """
    diabetic_data = pd.read_csv('data/raw_data/diabetic_data.csv')
    diabetic_data = DataEraser(diabetic_data).delete_unnecessary_data()
    diabetic_data = DataEncoder(diabetic_data).encode_data()

    diabetic_data.to_csv("data/processed_data/latest_for_clustering_modified_data.csv", index=False)


if __name__ == '__main__':
    main()
