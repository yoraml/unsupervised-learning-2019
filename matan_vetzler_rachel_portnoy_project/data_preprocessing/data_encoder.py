import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataEncoder:
    def __init__(self, diabetic_data):
        self.diabetic_data = diabetic_data

    def encode_data(self):
        """
        main function responsible to encode diabetic data after erasing specific data by custom and automatic methods
        :return: data set after encoding it's data
        """
        return self._custom_encode_categorical_data() \
            ._automatic_encode_categorical_data() \
            .diabetic_data

    def _custom_encode_categorical_data(self):
        """
        custom encoding diabetic data set
        :return: class object after custom encoding data
        """
        return self._encode_miglitol() \
            ._encode_medical_specialty() \
            ._encode_diagnosis_code() \
            ._encode_age() \
            ._encode_glucose_level() \
            ._encode_hemoglobin_level() \
            ._encode_medicines_change() \
            ._encode_readmitted() \
            ._encode_nominal_data_to_str()

    def _automatic_encode_categorical_data(self):
        """
        automatic encoding data set with one-hot encoder and normalizing with z-score
        :return: class object after automatic encoding data
        """
        scaler = StandardScaler()
        numeric_columns = self.diabetic_data._get_numeric_data().columns
        self.diabetic_data[numeric_columns] = scaler.fit_transform(self.diabetic_data[numeric_columns])
        categorical_columns = self._get_categorical_columns()
        self.diabetic_data = pd.get_dummies(self.diabetic_data, columns=list(categorical_columns))

        return self

    def _encode_miglitol(self):
        """
        function responsible for encoding miglitol feature which stands for medicine submission
        as seen people who were submitted with lower level of miglitol surely came back.
        :return: class object with diabetic data after applying encoding methods on
        """
        miglitol_dict = {'Up': 0,
                         'Down': 500,
                         'Steady': 0,
                         'No': 0
                         }
        self.diabetic_data['miglitol'] = self.diabetic_data['miglitol'].replace(miglitol_dict)

        return self

    def _encode_diagnosis_code(self):
        """
        function responsible for encoding column - diagnosis code by each code and it's corresponding value
        :return: class object after encoding diagnosis code column
        """
        self._medically_map_diagnosis()

        # diagnosis grouping
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols:
            self.diabetic_data['temp'] = np.nan

            condition = self.diabetic_data[col] == 250
            self.diabetic_data.loc[condition, 'temp'] = 'Diabetes'

            condition = self.diabetic_data[col] == 0
            self.diabetic_data.loc[condition, col] = '?'
            self.diabetic_data['temp'] = self.diabetic_data['temp'].fillna('Other')
            condition = self.diabetic_data['temp'] == '0'
            self.diabetic_data.loc[condition, 'temp'] = np.nan
            self.diabetic_data[col] = self.diabetic_data['temp']
            self.diabetic_data.drop('temp', axis=1, inplace=True)

        self.diabetic_data.dropna(inplace=True)

        return self

    def _medically_map_diagnosis(self):
        """
        function responsible to map diagnosis and their medical code
        :return: None
        """
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        for col in diag_cols:
            self.diabetic_data[col] = self.diabetic_data[col].str.replace('E', '-')
            self.diabetic_data[col] = self.diabetic_data[col].str.replace('V', '-')
            condition = self.diabetic_data[col].str.contains('250')
            self.diabetic_data.loc[condition, col] = '250'

        self.diabetic_data[diag_cols] = self.diabetic_data[diag_cols].astype(float)

    def _encode_age(self):
        """
        function responsible for encoding age column
        :return: class object after encoding age column
        """
        self.diabetic_data.loc[self.diabetic_data['age'] == '[0-10)', 'age'] = 5
        self.diabetic_data.loc[self.diabetic_data['age'] == '[10-20)', 'age'] = 15
        self.diabetic_data.loc[self.diabetic_data['age'] == '[20-30)', 'age'] = 25
        self.diabetic_data.loc[self.diabetic_data['age'] == '[30-40)', 'age'] = 35
        self.diabetic_data.loc[self.diabetic_data['age'] == '[40-50)', 'age'] = 45
        self.diabetic_data.loc[self.diabetic_data['age'] == '[50-60)', 'age'] = 55
        self.diabetic_data.loc[self.diabetic_data['age'] == '[60-70)', 'age'] = 65
        self.diabetic_data.loc[self.diabetic_data['age'] == '[70-80)', 'age'] = 75
        self.diabetic_data.loc[self.diabetic_data['age'] == '[80-90)', 'age'] = 85
        self.diabetic_data.loc[self.diabetic_data['age'] == '[90-100)', 'age'] = 95

        return self

    def _encode_glucose_level(self):
        """
        function responsible for encoding glucose levels in blood based on medical information
        :return: class object after encoding glucose column
        """
        max_glu_serum_dict = {'None': 0,
                              'Norm': 100,
                              '>200': 200,
                              '>300': 300
                              }
        self.diabetic_data['max_glu_serum'] = self.diabetic_data['max_glu_serum'].replace(max_glu_serum_dict)

        return self

    def _encode_hemoglobin_level(self):
        """
        function responsible for encoding HbA1c hemoglobin levels in blood based on medical information
        :return: class object after encoding HbA1c test result column
        """
        hba1c_dict = {'None': 0,
                      'Norm': 5,
                      '>7': 10,
                      '>8': 15
                      }
        self.diabetic_data['A1Cresult'] = self.diabetic_data['A1Cresult'].replace(hba1c_dict)

        return self

    def _encode_medicines_change(self):
        """
        function responsible for encoding change in medicines submission
        :return: class object after encoding medicines change column
         """
        medicines_change_dict = {'No': 0,
                                 'Ch': 1
                                 }
        self.diabetic_data['change'] = self.diabetic_data['change'].replace(medicines_change_dict)

        return self

    def _encode_diabetes_medicines_submission(self):
        """
        function responsible for encoding diabetic medicines submission
        :return: class object after encoding diabetic medicines submission column
        """
        diabetes_medicines_submission_dict = {'No': 0,
                                              'Yes': 1
                                              }
        self.diabetic_data['diabetesMed'] = self.diabetic_data['diabetesMed'].replace(
            diabetes_medicines_submission_dict)

        return self

    def _encode_medical_specialty(self):
        """
        function responsible for encoding medical specialty column based on 10 highest informative medical specialty
        :return: class object after encoding medical specialty column
        """
        top_ten_medical_speciality = ['?', 'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
                                      'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
                                      'Orthopedics-Reconstructive', 'Radiologist']

        self.diabetic_data['med_spec'] = self.diabetic_data['medical_specialty'].copy()
        self.diabetic_data.loc[~self.diabetic_data.medical_specialty.isin(
            top_ten_medical_speciality), 'med_spec'] = 'Other'
        self.diabetic_data.drop(['medical_specialty'], axis=1, inplace=True)

        return self

    def _encode_readmitted(self):
        readmitted_dict = {'NO': '0',
                           '>30': '1',
                           '<30': '2'
                           }
        self.diabetic_data['readmitted'] = self.diabetic_data['readmitted'].replace(readmitted_dict)

        return self

    def _encode_nominal_data_to_str(self):
        """
        function responsible to change nominal numeric columns to string values so won't be counted at z score
        normalization, but in one-hot encoding
        :return: class object after change in those nominal numeric columns
        """
        for column in ['discharge_disposition_id', 'admission_type_id', 'admission_source_id']:
            self.diabetic_data[column] = self.diabetic_data[column].astype(str)

        return self

    def _get_categorical_columns(self):
        """
        function responsible for retrieving all data categorical columns
        :return: data categorical columns
        """
        columns = self.diabetic_data.columns
        numeric_columns = self.diabetic_data._get_numeric_data().columns

        return set(columns) - set(numeric_columns) - {'readmitted'}
