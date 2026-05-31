import math
from features import get_rid_fake_zero
import pandas as pd
# import numpy as np

df = pd.read_excel('../data/raw/Diabetes data CLAZIQ.xlsx')


# Columns without fake zeros
age = df["Age"]
Diabetes_pedigree_function = df["DiabetesPedigreeFunction"]
bmi = df["BMI"]
pregnancies = df["Pregnancies"]

# Columns with fake zeros
glucose = df["Glucose"]
blood_pressure = df["BloodPressure"]
skin_thickness = df["SkinThickness"]
insulin = df["Insulin"]
outcome = df["Outcome"]
# df
type(df)

# Parsing the columns into the function
glucose = get_rid_fake_zero(df, glucose)
glucose = replace_zero(df, glucose)


def replace_zero(df, column):
    for col in column:
        str(col)
        for i in col:
            if i == 0:
                i = math.fsum(column) / 768
    return column


# variable_predictor = [age, bmi, glucose, pregnancies, blood_pressure,
#                       skin_thickness, insulin, Diabetes_pedigree_function]
np.dot(age, insulin)
print(insulin[45])
print(type(insulin))
outcome.sum()
insulin.sum()
df


def summ(column):
    for number in column:
        summed = number ** 2
        summer = summed.sum()
        # print(f"summed insulin squared {summer}")
    return summer


summ('insulin')
