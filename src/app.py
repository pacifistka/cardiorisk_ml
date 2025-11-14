import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI
from pydantic import BaseModel
from src.model import HeartRiskModel

app = FastAPI(title="Heart Risk API")
model = HeartRiskModel()


class PredictRequest(BaseModel):
    csv_path: str


@app.post("/predict")
def predict(req: PredictRequest):
    df = model.predict_from_csv(req.csv_path)
    return {
        "n_samples": len(df),
        "predictions": df.to_dict(orient="records")
    }