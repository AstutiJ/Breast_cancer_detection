import pandas as pd
import numpy as np
data = pd.read_csv(r'C:\Users\astut\Desktop\Breast cancer detection\data.csv')
print(data.head(1))
print(data.shape)
print(data.info())
print(data.columns)

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
y = data[['diagnosis']]

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


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

model.fit(x,y)

Y_predicted= model.predict(x)
print("x_predict y: ", Y_predicted)

acc = model.score(x,y)
print('Acc',acc)

Y_predicted=model.predict(sample_data)
print("X_predicted y",Y_predicted)


import joblib as jb
#jb.dump(Name of trained model,name)

jb.dump(model,'Breast_cancer_KNN.pkl')
print("model saved")
# model saved
# to save the model 