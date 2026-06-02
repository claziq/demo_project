import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_cleaning():
    # Loading the dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, '..', 'data', 'raw',
                             'Diabetes data CLAZIQ.xlsx')
    diabetes_df = pd.read_excel(file_path)

    # Columns are loaded and made into pandas dataframe
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI', 'Outcome']
    diabetes_df = pd.DataFrame(diabetes_df, columns=columns)

    # Converting the 0s into column mean for the column with impossible 0s
    columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']

    for col in columns_to_fix:
        diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
        column_mean = diabetes_df[col].mean()
        diabetes_df[col] = diabetes_df[col].fillna(column_mean)

    # Standardizing each column using Z-score
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure',
                    'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI']

    # Splitting my dataset to seperate the target from the features
    x_df = diabetes_df[feature_cols]
    y_df = diabetes_df['Outcome']

    # Converting the pandas dataframe into numpy array
    x = x_df.values
    y = y_df.values

    # test data 20% and train data 80%
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.2, random_state=84)

    # Applying feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled, X_test_scaled, Y_train, Y_test
