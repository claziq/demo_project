import numpy as np
import pandas as pd


df = pd.read_excel('data/raw/Diabetes data CLAZIQ.xlsx')


def get_rid_fake_zero(df, column_name):
    main_column_data = df[column_name].to_numpy()
    main_column_data = np.where(
        main_column_data == 0, np.nan, main_column_data)
    column_mean = np.nanmean(main_column_data)
    main_column_data[np.isnan(main_column_data)] = column_mean
    df[column_name] = main_column_data
    return main_column_data


column_to_mean = ["BMI", "Glucose", "Pregnancies", "BloodPressure",
                  "SkinThickness", "Insulin", "DiabetesPedigreeFunction"]

for col in column_to_mean:
    df[col] = get_rid_fake_zero(df, col)
