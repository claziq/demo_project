import os
from features import get_rid_fake_zero
import pandas as pd
import numpy as np


df = pd.read_excel('../data/raw/Diabetes data CLAZIQ.xlsx')
# df.head()


column_to_mean = [df.BMI, df.Glucose, df.Pregnancies, df.BloodPressure,
                  df.SkinThickness, df.Insulin, df.DiabetesPedigreeFunction]
column_to_mean
type(df['Age'])
df['Age'][700]
df.at[698, 'Age']
df.Age
for col in column_to_mean:
    df[col] = get_rid_fake_zero(df, col)
    df[col].to_numpy()


# bmi = get_rid_fake_zero(df, "BMI")
# glucose = get_rid_fake_zero(df, "Glucose")
# pregnancies = get_rid_fake_zero(df, "Pregnancies")
# blood_pressure = get_rid_fake_zero(df, "BloodPressure")
# skin_thickness = get_rid_fake_zero(df, "SkinThickness")
# insulin = get_rid_fake_zero(df, "Insulin")
# Diabetes_pedigree_function = get_rid_fake_zero(df, "DiabetesPedigreeFunction")
age = df["Age"].to_numpy()


feature_columns = np.array[[df['BMI'], df['Glucose'], df['Pregnancies'], df['BloodPressure'],
                            df['SkinThickness'], df['Insulin'], df['DiabetesPedigreeFunction'], df['Age']]]

exp = df['Age'] + df['Insulin']
exp.describe()

feature_columns

X = df[[column_to_mean]]
Y = df[["Outcome"]]


with pd.ExcelWriter('feature_columns.xlsx', engine='openpyxl') as writer:
    feature_data = pd.DataFrame(X)
    predict_data = pd.DataFrame(Y)

X.to_excel('feature_data.xlsx', index=False, sheet_name='Dataset1')
Y.to_excel('predict_data.xlsx', index=False, sheet_name='Dataset2')


print(X)
df.loc[764]
df.loc[500:550]
df.tail(10)
df.sample(10)


confirmed_diabetes = df.Outcome.sum()
print(f"Total number of people with Diabetes are {confirmed_diabetes}")


for body in df.BMI:
    count = []
    if body >= df.BMI.mean():
        count += 1
    print(
        f"Total number of people with BMI higher than {df.BMI.mean()} is {count}")
