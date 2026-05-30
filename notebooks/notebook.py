import os
from features import get_rid_fake_zero
import pandas as pd
import numpy as np


df = pd.read_excel('../data/raw/Diabetes data CLAZIQ.xlsx')
# df.head()


column_to_mean = ["BMI", "Glucose", "Pregnancies", "BloodPressure",
                  "SkinThickness", "Insulin", "DiabetesPedigreeFunction"]

for col in column_to_mean:
    df[col] = get_rid_fake_zero(df, col)


bmi = get_rid_fake_zero(df, "BMI")
glucose = get_rid_fake_zero(df, "Glucose")
pregnancies = get_rid_fake_zero(df, "Pregnancies")
blood_pressure = get_rid_fake_zero(df, "BloodPressure")
skin_thickness = get_rid_fake_zero(df, "SkinThickness")
insulin = get_rid_fake_zero(df, "Insulin")
Diabetes_pedigree_function = get_rid_fake_zero(df, "DiabetesPedigreeFunction")
age = df["Age"].to_numpy()


feature_columns = ["BMI", "Glucose", "Pregnancies", "BloodPressure",
                   "SkinThickness", "Insulin", "DiabetesPedigreeFunction", "Age"]
X = feature_columns
Y = df["Outcome"]


feature_data = pd.DataFrame(X)
predict_data = pd.DataFrame(Y)

feature_data.to_excel('feature_data.xlsx', index=False, sheet_name='Dataset1')
feature_data.to_excel('predict_data.xlsx', index=False, sheet_name='Dataset2')


print(X)
df
