from fastapi import FastAPI, Request
import uvicorn
import pandas as pd
from sklearn.externals import joblib

app = FastAPI()

# Load your trained model
model = joblib.load("model.pkl")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": pred.tolist()}
