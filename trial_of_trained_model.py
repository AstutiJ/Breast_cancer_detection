import numpy as np
import pandas as pd
import joblib as jb
model=jb.load('Breast_cancer_KNN.pkl')
print('Model loaded successfully')
sample_data = pd.DataFrame({
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
Y_predicted=model.predict(sample_data)
print("X_predicted y",Y_predicted)

if (Y_predicted==0):
    print("Congraluations you don't have cancer.")
else:
    print("you have breast cancer. visit to doctor")