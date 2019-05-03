import numpy as np


class DataEraser:
    def __init__(self, diabetic_data):
        self.diabetic_data = diabetic_data

    def delete_unnecessary_data(self):
        """
        function uniting all deletion manipulation techniques on the diabetic data
        :return: class object with diabetic data after applying deletion methods on
        """
        return self._delete_duplicated_patient_nbr() \
            ._delete_uninformative_codes_data() \
            ._delete_gender_column() \
            ._delete_high_percentage_missing_data_columns() \
            ._delete_missing_data_rows() \
            ._delete_specific_medications_columns() \
            .diabetic_data

    def _delete_uninformative_codes_data(self):
        """
        deleting columns with id information which don't provide useful
        information clustering wise
        :return: class object with diabetic data after uninformative id data deletion
        """

        # deleting unique elements column - encounter id and patient number, as don't provide useful
        # information clustering wise
        self.diabetic_data.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

        # deleting discharge disposition id elements which relates for death or hospice
        # as described at IDs_mapping.csv file, since those patients won't be able to be readmitted
        death_hospice_codes = [11, 13, 14, 19, 20, 21]
        self.diabetic_data = \
            self.diabetic_data.loc[~self.diabetic_data.discharge_disposition_id.isin(death_hospice_codes)]

        return self

    def _delete_high_percentage_missing_data_columns(self):
        """
        deleting columns with high level of missing data percentage
        :return: class object with diabetic data after high level of missing data percentage columns deletion
        """

        # weight column missing rate : 97%
        # payer_code missing rate: 52%
        self.diabetic_data.drop(['weight', 'payer_code'], axis=1, inplace=True)

        return self

    def _delete_gender_column(self):
        """
        deleting gender columns due to approx same number of male-female patients ration,
        with approx same percentages of readmitted levels
        :return: class object with diabetic data after gender column deletion
        """
        self.diabetic_data.drop(['gender'], axis=1, inplace=True)

        return self

    def _delete_missing_data_rows(self):
        """
        deleting lines with missing features data
        :return: class object with diabetic data after rows with missing features data deletion
        """

        # after deleting high level of missing data percentage columns
        # deleting few of the lines containing missing values represented in out
        # data set as - ["?", "Unknown/Invalid"]
        self.diabetic_data = self.diabetic_data.replace(['?', 'Unknown/Invalid'], np.nan)
        all_columns = list(self.diabetic_data.columns.values)
        all_columns.remove('medical_specialty')
        self.diabetic_data.dropna(inplace=True, subset=all_columns)

        return self

    def _delete_specific_medications_columns(self):
        """
        deleting specific medications columns
        :return: class object with diabetic data after specific medications columns
        """

        # the following medicines were provided to extremely small amount of hospitalized
        # patients (1 - 10), thus cannot learned from, due to low level of documentations
        # around 100K examples
        low_versatile_columns = ['acetohexamide', 'citoglipton', 'examide', 'glimepiride-pioglitazone',
                                 'metformin-pioglitazone', 'metformin-rosiglitazone', 'troglitazone', 'acarbose']
        self.diabetic_data.drop(low_versatile_columns, axis=1, inplace=True)

        # the following medicines do not provide any useful or interesting information thus being deleted
        useless_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide',
                           'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'tolazamide',
                           'glyburide-metformin', 'glipizide-metformin']
        self.diabetic_data.drop(useless_columns, axis=1, inplace=True)

        return self

    def _delete_duplicated_patient_nbr(self):
        """
        deleting duplicated same patient inpatient/outpatient occurrence
        :return: class object with diabetic data after patient number duplicate deletion
        """
        self.diabetic_data.drop_duplicates(subset='patient_nbr', inplace=True)

        return self
