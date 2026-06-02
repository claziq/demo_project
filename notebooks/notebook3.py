from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# setting a random seed
np.random.seed(42)

# Loading the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, '..', 'data', 'raw',
                         'Diabetes data CLAZIQ.xlsx')

# Columns are loaded and made into pandas dataframe
diabetes_df = pd.read_excel(file_path)
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI', 'Outcome']
diabetes_df = pd.DataFrame(diabetes_df, columns=columns)

# List of columns with 0s in them which is not possible(medically)
# columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']

# for col in columns_with_zeros:
#     diabetes_df.loc[diabetes_df.sample(frac=0.1).index, col] = 0
# print("Data loaded and prepared")


# Converting the 0s into column mean for the column with impossible 0s
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']

for col in columns_to_fix:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
    column_mean = diabetes_df[col].mean()
    diabetes_df[col] = diabetes_df[col].fillna(column_mean)
print("Zeros replaced with column mean.")

# Standardizing each column using Z-score
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure',
                'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI']

# for col in feature_cols:
#     diabetes_df[col] = (diabetes_df[col] -
#                         diabetes_df[col].mean() / diabetes_df[col].std())

# Splitting my dataset to seperate the target from the features
x_df = diabetes_df.iloc[:, :7]
y_df = diabetes_df.iloc[:, 8:9]

# Writing the split into seperate files
# x_df.to_excel('features_data.xlsx', index=False)
# y_df.to_excel('target_data.xlsx', index=False)
print("Feature and target saved to different file.")

# Converting the pandas dataframe into numpy array
x = x_df.values
y = y_df.values

# Splitted the test data 20% and train data 80%
indices = np.random.permutation(len(x))
test_size = int(len(x) * 0.2)
test_indices = indices[: test_size]
train_indices = indices[test_size:]

# test data 20% and train data 80%
X_train, X_test = x[train_indices], x[test_indices]
Y_train, Y_test = y[train_indices], y[test_indices]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Adding the intercept
X_train_b = np.c_[np.ones((len(X_train_scaled), 1)), X_train_scaled]
X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]

# Fitting the model to normal equation
best_fit_theta = np.linalg.inv(X_train_b.T.dot(
    X_train_b)).dot(X_train_b.T).dot(Y_train)

# pridicting with the model by multiplying X_test_b with best_fit_theta
predictions = X_test_b.dot(best_fit_theta)

# Calculating the mean square error
mse = np.mean((Y_test - predictions) ** 2)


print("Regression model trained from scratch")
print(f"-> Calculated the coefficient: {best_fit_theta}")
print(f"-> Mean Square Error: {mse:.3f}")


# logit beginning
print("This is the beginning of Logistic Regression")


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]


# setting hyperparameters for gradient descent
learning_rate = 0.001
iterations = 5000
Y_train = Y_train.reshape(-1)
m = len(Y_train)

# Initializing the coefficient to zero
theta = np.zeros(X_train_b.shape[1])


# Loop to upgrade coefficients gradually
for i in range(iterations):
    # calculating the linear steps, then applying the segmoid function
    linear_model = np.dot(X_train_b, theta)
    predictions = sigmoid(linear_model)
    # calculating the gradient
    # (1/m) * X^T * (predictions - actual). the formula calculating the gradient descent
    gradient = np.dot(X_train_b.T, (predictions - Y_train)) / m

    # we have to update the coefficient by taking a step in the opposite direction of the gradinet
    theta -= learning_rate * gradient


# get the probability for the test
test_linear_model = np.dot(X_test_b, theta)
test_probabilities = sigmoid(test_linear_model)

# convert probabilities to strict 0 and 1 predictions
final_prediction = (test_probabilities >= 0.5).astype(int)

# checking for the acuracy using Accuracy(percentage correct)
accuracy = np.mean(final_prediction == Y_test)

print("Logit regression model trained via Gradient descent.")
print(f"-> Calculated Coefficients {theta}")
print(f"-> Model Accuracy {accuracy * 100:.3f}%")


# Random forest model
print("This is the beginning of random forest!")


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

best_randfor_model = GridSearchCV(
    estimator=RandomForestClassifier(random_state=84),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1
)

# randfor_model = RandomForestClassifier(n_estimators=100, random_state=84)
best_randfor_model.fit(X_train, Y_train.flatten())

best_tuned_params = best_randfor_model.best_params_

best = best_randfor_model.best_estimator_
randfor_predictions = best_randfor_model.predict(X_test)

randfor_accuracy = accuracy_score(Y_test.flatten(), randfor_predictions) * 100

print(f"Best Parameters: {best_tuned_params}")
print(f"Random Forest Model Accuracy: {randfor_accuracy:.3f}%")


print("This is the beginning of gradient boosting")
grad_boosting_model = HistGradientBoostingClassifier(
    random_state=84, max_iter=100)

grad_boosting_model.fit(X_train, Y_train.flatten())


grad_boost_predict = grad_boosting_model.predict(X_test)
grad_boost_accuracy = accuracy_score(
    Y_test.flatten(), grad_boost_predict) * 100

print(f"Gradient Boosting Accuracy Score: {grad_boost_accuracy:.3f}%")
