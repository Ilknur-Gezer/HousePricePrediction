import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

# Load training data
training = pd.read_csv('/Users/igezer/Desktop/house-prices-advanced-regression-techniques/train.csv')
X = training.drop(columns=['Id', 'SalePrice'])
y = training['SalePrice']

# Split into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit-transform training and transform validation data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)

# Try different models and print RMSEs
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(random_state=42)
}

best_model = None
best_rmse = float('inf')

for name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    y_pred = model.predict(X_val_preprocessed)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"{name} Validation RMSE: {rmse:.2f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

print("\nBest model based on validation RMSE:", type(best_model).__name__)

# Train best model (XGBoost) on full training data
X_full_preprocessed = preprocessor.fit_transform(X)
final_model = xgb.XGBRegressor(random_state=42, verbosity=0)
final_model.fit(X_full_preprocessed, y)

# Load and preprocess test data
test = pd.read_csv('/Users/igezer/Desktop/house-prices-advanced-regression-techniques/test.csv')
test_ids = test['Id']
X_test = test.drop(columns=['Id'])
X_test_preprocessed = preprocessor.transform(X_test)

# Predict and save results
y_test_pred = final_model.predict(X_test_preprocessed)
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': y_test_pred})
submission.to_csv('/Users/igezer/Desktop/house-prices-advanced-regression-techniques/test_with_predictions.csv', index=False)
print("\nPredictions saved to test_with_predictions.csv")


# Load training and test prediction data
train = pd.read_csv("/Users/igezer/Desktop/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/Users/igezer/Desktop/house-prices-advanced-regression-techniques/test_with_predictions.csv")

# Extract SalePrice from training data and predictions from test data
actual_prices = train['SalePrice'].values
predicted_prices = test['SalePrice'].values

# Plot
plt.figure(figsize=(10,6))
plt.hist(actual_prices, bins=30, color='lightpink', edgecolor='black', label='Training SalePrice', alpha=0.6)
plt.hist(predicted_prices, bins=30, color='steelblue', edgecolor='black', label='Predicted SalePrice', alpha=0.7)

plt.title('Sale Price Distribution: Training vs Predicted')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.legend()
plt.show()