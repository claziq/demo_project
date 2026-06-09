from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
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


# Converting the 0s into column mean for the column with impossible 0s
columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']

for col in columns_to_fix:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
    column_mean = diabetes_df[col].mean()
    diabetes_df[col] = diabetes_df[col].fillna(column_mean)
print("Zeros replaced with column mean.")

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure',
                'Insulin', 'DiabetesPedigreeFunction', 'Age', 'BMI']


# Splitting my dataset to seperate the target from the features and converting to numpy arrays
x = diabetes_df.drop(columns=['Outcome']).values
y = diabetes_df['Outcome'].values


# Splitted the test data 20% and train data 80%
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=84)

# using decision tree
print("--- This is the begining of Decision Tree Classifier ---")
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
score = accuracy_score(Y_test, prediction)
print(f"This is the Accuracy Score for Decision Tree {score}")

# Scaling the input features down to better match with the outcome variable
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# This is the beginning of Linear Regression
print("--- Regression Begins ---")
model_reg = LinearRegression()
model_reg.fit(X_train_scaled, Y_train)
prediction_reg = model_reg.predict(X_test_scaled)
mse = mean_squared_error(Y_test, prediction_reg)

print(f"-> Mean Square Error: {mse:.3f}")
print(f"R-square score: {r2_score(Y_test, prediction_reg):.3f}")
print(f"Coefficients: {model_reg.coef_}")
print(f"Intercept: {model_reg.intercept_}")


# logit beginning
print("--- This is the beginning of Logistic Regression ---")

log_model = LogisticRegression(class_weight='balanced')
log_model.fit(X_train_scaled, Y_train)
log_predict = log_model.predict(X_test_scaled)
print(f"Predicted classes: {log_predict}")
log_score = accuracy_score(Y_test, log_predict)
print(f"logistics scores: {log_score:.3f}")
print("\nDetailed Report:\n", classification_report(Y_test, log_predict))
print(f"-> Model Coefficient {log_model.coef_}")


# Random forest model
print("--- This is the beginning of random forest! ---")


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


print("--- This is the beginning of gradient boosting ---")
grad_boosting_model = HistGradientBoostingClassifier(
    random_state=84, max_iter=100)

grad_boosting_model.fit(X_train, Y_train.flatten())


grad_boost_predict = grad_boosting_model.predict(X_test)
grad_boost_accuracy = accuracy_score(
    Y_test.flatten(), grad_boost_predict) * 100

print(f"Gradient Boosting Accuracy Score: {grad_boost_accuracy:.3f}%")
