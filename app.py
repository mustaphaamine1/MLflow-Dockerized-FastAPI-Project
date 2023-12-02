
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
import onnxruntime

app = FastAPI()

dataset = pd.read_csv('kc_house_data.csv')
dataset.drop(columns=['id', 'date', 'sqft_lot', 'condition', 'zipcode', 'long', 'sqft_lot15'], inplace=True)
X = dataset.drop(columns='price')
print(X.head())

model = mlflow.sklearn.load_model("/app/mlruns/489022706431852524/5e6eff70ee23497f98aab6759bf10848/best_model")
if model:
    print("Model loaded successfully.")
else:
    print("Error: Failed to load the model.")


@app.post('/predict')
async def predict(input_data: dict):
    try:
        input_list = [input_data]
        input_df = pd.DataFrame(input_list)
        predictions = model.predict(input_df)
        print('good')
        return predictions.tolist()
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))


'''
{
    "bedrooms": 3,
    "bathrooms": 1.00,
    "sqft_living": 1180,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "grade": 7,
    "sqft_above": 1180,
    "sqft_basement": 0,
    "yr_built": 1955,
    "yr_renovated": 0,
    "lat": 47.5112,
    "sqft_living15": 1340
}
'''