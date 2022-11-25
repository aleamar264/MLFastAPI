import gzip

import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# load data
data = pd.read_csv('mlfastapi1/data/breast_cancer.csv')

# Preselected the features
selected_features = [
    'concavity_mean',
    'concave_points_mean',
    'perimeter_se',
    'area_se',
    'texture_worst',
    'area_worst'
]

# Preprocess data
data = data.set_index('id')
data['diagnosis'] = data['diagnosis'].replace(['B', 'M'], [0, 1]) # Encode y, B -> 0 , M -> 1

# split data 80% - 20%
y = data.pop('diagnosis')
X = data
X = X[selected_features.copy()]
X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# create an ensemble of 3 models
estimators = []
estimators.append(('logistic', LogisticRegression()))
estimators.append(('cart', DecisionTreeClassifier()))
estimators.append(('svm', SVC()))

# create the ensemble model
ensemble = VotingClassifier(estimators)

# Make preprocessor Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer()), # Missing value imputer
    ('scaler', MinMaxScaler(feature_range=(0,1))),
    ('model', ensemble)
])

# train model
pipe.fit(X_train, y_train)

# Test accuracy
print(f"Accuraccy: {str(round(pipe.score(X_Test, y_test), 3) * 100)}")

# Export model
joblib.dump(pipe, gzip.open('mlfastapi1/model/model_binary_data.gz', "wb"))