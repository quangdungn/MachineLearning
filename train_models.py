import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Đọc dữ liệu
df = pd.read_csv('housing.csv')

# Xử lý giá trị thiếu
df['total_bedrooms'].fillna(df['total_bedrooms'].mean(), inplace=True)

# Chia dữ liệu thành X và y
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Xác định các cột số và cột phân loại
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                    'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['ocean_proximity']

# Tạo bộ tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm tạo pipeline
def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

# Huấn luyện mô hình Linear Regression
lr_pipeline = create_pipeline(LinearRegression())
lr_pipeline.fit(X_train, y_train)

# Huấn luyện mô hình Ridge Regression
ridge_pipeline = create_pipeline(Ridge(alpha=1.0))
ridge_pipeline.fit(X_train, y_train)

# Huấn luyện mô hình Neural Network (MLPRegressor)
mlp_pipeline = create_pipeline(MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
mlp_pipeline.fit(X_train, y_train)

# Huấn luyện mô hình Stacking Regressor
estimators = [
    ('lr', LinearRegression()),
    ('ridge', Ridge(alpha=1.0)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
]

stacking_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge()
    ))
])
stacking_pipeline.fit(X_train, y_train)

# Tạo thư mục 'models' nếu chưa tồn tại
if not os.path.exists('models'):
    os.makedirs('models')

# Lưu các mô hình
joblib.dump(lr_pipeline, 'models/linear_regression_model.joblib')
joblib.dump(ridge_pipeline, 'models/ridge_regression_model.joblib')
joblib.dump(mlp_pipeline, 'models/mlp_regressor_model.joblib')
joblib.dump(stacking_pipeline, 'models/stacking_regressor_model.joblib')

print("Huấn luyện và lưu mô hình thành công!")
