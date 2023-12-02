import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature
import joblib

dataset = pd.read_csv('kc_house_data.csv')

def preprocess_data(dataset):
    dataset.drop(columns=['id', 'date', 'sqft_lot', 'condition', 'zipcode', 'long', 'sqft_lot15'], inplace=True)
    X = dataset.drop(columns='price')
    y = dataset.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(dataset)

best_model = None
best_mse = float('inf')

algorithms = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(max_iter=2000),
}

mlflow.set_experiment("my_experiment")

for algo_name, algo in algorithms.items():
    with mlflow.start_run():
        mlflow.log_param("algorithm", algo_name)
        algo.fit(X_train, y_train)
        predictions = algo.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(algo, "model")

        if mse < best_mse:
            best_mse = mse
            best_model = algo

# Save the best model separately
mlflow.sklearn.log_model(best_model, "best_model")

# Save preprocessing transformations using Pickle format
joblib.dump(preprocess_data, "preprocessing_transformations.pkl")
'''
# Save the best model in ONNX format
input_test = X_test.iloc[:1]
onnx_path = "best_model.onnx"
mlflow.onnx.log_model(best_model, onnx_path, input_example=input_test)
'''
import numpy as np
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import onnx


feature_count = X_test.shape[1]
input_data_type = [('input', FloatTensorType([None, feature_count]))]

onnx_model = convert_sklearn(best_model, initial_types=input_data_type)
onnx.save_model(onnx_model, 'best_model.onnx')