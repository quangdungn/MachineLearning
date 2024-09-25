from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Tải mô hình và bộ tiền xử lý
lr_model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/linear_regression_model.joblib'))
ridge_model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/ridge_regression_model.joblib'))
mlp_model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/mlp_regressor_model.joblib'))
stacking_model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/stacking_regressor_model.joblib'))

# Tạo dictionary cho các mô hình
models = {
    'Linear Regression': lr_model,
    'Ridge Regression': ridge_model,
    'Neural Network': mlp_model,
    'Stacking Regressor': stacking_model
}

# Trang chủ với form nhập liệu
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        data = request.form.to_dict()
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame([data])
        # Chuyển đổi các giá trị về dạng số (float)
        numeric_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                           'total_bedrooms', 'population', 'households', 'median_income']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        # Kiểm tra giá trị thiếu
        if df[numeric_columns].isnull().any().any():
            return "Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra lại."
        # Sắp xếp các cột theo đúng thứ tự
        df = df[models['Linear Regression'].named_steps['preprocessor'].transformers_[0][2] + ['ocean_proximity']]
        # Lấy tên mô hình được chọn
        model_name = request.form.get('model')
        # Lấy mô hình tương ứng
        model = models.get(model_name)
        # Dự đoán kết quả
        prediction = model.predict(df)[0]
        # Lấy thông tin độ tin cậy
        model_scores = {
            'Linear Regression': 69043.17,
            'Ridge Regression': 69043.17,
            'Neural Network': 56023.45,
            'Stacking Regressor': 55012.34
        }
        confidence = model_scores.get(model_name)
        return render_template('result.html', prediction=prediction, confidence=confidence, model_name=model_name)
    else:
        return render_template('index.html')

# Điểm vào cho Vercel
def handler(event, context):
    from vercel_wsgi import handle_wsgi
    return handle_wsgi(app, event, context)
