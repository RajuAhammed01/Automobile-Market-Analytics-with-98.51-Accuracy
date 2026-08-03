import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor 
import joblib

print("Loading dataset...")
df = pd.read_csv('automobile_dataset.csv')

# Handle missing values
num_cols = ['Engine_Size', 'Horsepower', 'Torque', 'Accident_History', 'Fuel_Efficiency']
cat_cols = ['Transmission', 'Service_History', 'Color', 'Location']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

num_cols_pipe = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols_pipe = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols_pipe),
        ('cat', categorical_transformer, cat_cols_pipe)
    ])

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

param_distributions = {
    'regressor__n_estimators': [200, 300],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [4, 5],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_distributions,
    n_iter=5,
    cv=3, 
    scoring='r2',
    verbose=1,
    random_state=42,
    n_jobs=1
)

print("Starting Hyperparameter Tuning & Training...")
random_search.fit(X, y)

best_model = random_search.best_estimator_
print(f"Best Hyperparameters Found: {random_search.best_params_}")

model_filename = 'car_price_prediction_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved successfully to {model_filename}")
