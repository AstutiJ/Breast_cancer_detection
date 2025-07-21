
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score

# Load the dataset

data=pd.read_csv(r"C:\Users\astut\Desktop\Breast cancer detection\data.csv")
# Prepare features and target variable
x = data[[
    'radius_mean', 'texture_mean', 'perimeter_mean', 
    'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]]

# Convert diagnosis to numeric (assuming 'M' = 1 and 'B' = 0)
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Initialize models
model_LR = LinearRegression()
model_L = Lasso(alpha=0.99)
model_R = Ridge(alpha=0.99)

# Fit models
model_LR.fit(x, y)
model_L.fit(x, y)
model_R.fit(x, y)

# Predictions
Y_prediction_LR = model_LR.predict(x)
Y_prediction_L = model_L.predict(x)
Y_prediction_R = model_R.predict(x)

# Print predictions
print("Linear Regression Predictions:", Y_prediction_LR)
print("Lasso Predictions:", Y_prediction_L)
print("Ridge Predictions:", Y_prediction_R)

# Test data should match the feature set
test_data = pd.DataFrame({
    'radius_mean': [14.0],
    'texture_mean': [20.0],
    'perimeter_mean': [90.0],
    'area_mean': [600.0],
    'smoothness_mean': [0.10],
    'compactness_mean': [0.15],
    'concavity_mean': [0.12],
    'concave points_mean': [0.07],
    'symmetry_mean': [0.18],
    'fractal_dimension_mean': [0.06],
    'radius_se': [0.5],
    'texture_se': [1.0],
    'perimeter_se': [3.0],
    'area_se': [30.0],
    'smoothness_se': [0.006],
    'compactness_se': [0.02],
    'concavity_se': [0.02],
    'concave points_se': [0.01],
    'symmetry_se': [0.02],
    'fractal_dimension_se': [0.002],
    'radius_worst': [16.0],
    'texture_worst': [25.0],
    'perimeter_worst': [105.0],
    'area_worst': [800.0],
    'smoothness_worst': [0.12],
    'compactness_worst': [0.25],
    'concavity_worst': [0.30],
    'concave points_worst': [0.12],
    'symmetry_worst': [0.25],
    'fractal_dimension_worst': [0.07]
})

# Predictions on test data
test_result_1 = model_LR.predict(test_data)
test_result_2 = model_L.predict(test_data)
test_result_3 = model_R.predict(test_data)

# Print test results
print("model_LR : ", test_result_1)
print("model_L : ", test_result_2)
print("model_R : ", test_result_3)

# Model performance
acc_LR = r2_score(y, Y_prediction_LR)
print("model performance_LR: ", acc_LR)

acc_L = r2_score(y, Y_prediction_L)
print("model performance_L: ", acc_L)

acc_R = r2_score(y, Y_prediction_R)
print("model performance_R: ", acc_R)