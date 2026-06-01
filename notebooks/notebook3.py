import pandas as pd
import numpy as np

# setting a random seed
np.random.seed(42)

# Loading the dataset
diabetes_df = pd.read_excel('../data/raw/Diabetes data CLAZIQ.xlsx')

# Columns are loaded and made into pandas dataframe
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI', 'Outcome']
diabetes_df = pd.DataFrame(diabetes_df, columns=columns)

# List of columns with 0s in them which is not possible(medically)
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']

# for col in columns_with_zeros:
#     diabetes_df.loc[diabetes_df.sample(frac=0.1).index, col] = 0
# print("Data loaded and prepared")

# Standardizing each column
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']

for col in columns_to_fix:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
    column_mean = diabetes_df[col].mean()
    diabetes_df[col] = diabetes_df[col].fillna(column_mean)
print("Zeros replaced with column mean.")

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI', 'Outcome']
for col in feature_cols:
    diabetes_df[col] = (diabetes_df[col] -
                        diabetes_df[col].mean() / diabetes_df[col].std())


x_df = diabetes_df.iloc[:, :8]
y_df = diabetes_df.iloc[:, 8:9]


x_df.to_excel('features_data.xlsx', index=False)
y_df.to_excel('target_data.xlsx', index=False)
print("Feature and target saved to different file.")


x = x_df.values
y = y_df.values


indices = np.random.permutation(len(x))
test_size = int(len(x) * 0.2)

test_indices = indices[: test_size]
train_indices = indices[test_size:]

X_train, X_test = x[train_indices], x[test_indices]
Y_train, Y_test = y[train_indices], y[test_indices]


X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

best_fit_theta = np.linalg.inv(X_train_b.T.dot(
    X_train_b)).dot(X_train_b.T).dot(Y_train)


predictions = X_test_b.dot(best_fit_theta)

mse = np.mean((Y_test - predictions) ** 2)


print("Regression model trained from scratch")
print(f"-> Calculated the coefficient: {best_fit_theta}")
print(f"-> Mean Square Error: {mse:.2f}")
