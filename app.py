from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Tải các mô hình
model_dir = os.path.join(os.path.dirname(__file__), 'models')
lr_model = joblib.load(os.path.join(model_dir, 'linear_regression_model.joblib'))
ridge_model = joblib.load(os.path.join(model_dir, 'ridge_regression_model.joblib'))
mlp_model = joblib.load(os.path.join(model_dir, 'mlp_regressor_model.joblib'))
stacking_model = joblib.load(os.path.join(model_dir, 'stacking_regressor_model.joblib'))

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
        if df.isnull().any().any():
            return "Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra lại."
        
        # Lấy tên mô hình được chọn
        model_name = request.form.get('model')
        
        # Lấy mô hình tương ứng
        model = models.get(model_name)
        
        # Dự đoán kết quả
        try:
            prediction = model.predict(df)[0]
        except Exception as e:
            return f"Lỗi trong quá trình dự đoán: {e}"
        
        # Lấy thông tin độ tin cậy (ví dụ: RMSE trên tập kiểm tra)
        # Đây là giá trị giả định, bạn nên tính RMSE thực tế từ quá trình huấn luyện
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

if __name__ == '__main__':
    app.run(debug=True)
