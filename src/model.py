from pathlib import Path
import json
import joblib
import pandas as pd


class HeartRiskModel:
    def __init__(self):

        # путь к корню проекта
        root_path = Path(__file__).resolve().parents[1]

        artifacts_path = root_path / "artifacts"

        self.pipeline = joblib.load(artifacts_path / "pipeline.joblib")

        with open(artifacts_path / "meta.json") as f:
            meta = json.load(f)

        self.threshold = float(meta["threshold"])
        self.leakage_features = meta["leakage_features"]
        self.technical_features = meta["technical_features"]

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        # выбрасываем те же утечки и технические колонки, что в ноутбуке
        drop_cols = list(self.leakage_features) + list(self.technical_features)
        X = df.drop(columns=drop_cols, errors="ignore")
        return X

    def predict_from_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        X = self._prepare_X(df)
        proba = self.pipeline.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        # отдаёт DataFrame с колонками "id" и "prediction"
        result = pd.DataFrame({
            "id": range(len(preds)),
            "prediction": preds
        })
        return result
