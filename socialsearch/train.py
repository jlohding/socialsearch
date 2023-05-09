import joblib
import pandas as pd
from config_loader import config
from models import SponsorClfXGBPipeline

class Trainer:
    def __init__(self, model_pipeline):
        self.data = pd.read_pickle(config.datasets["data_path"])
        self.model_pipeline = model_pipeline

    def fit_model(self, X, y):
        self.model_pipeline.fit(X,y)
        self.model_pipeline.evaluate()
        return self

    def export_model(self, name):
        joblib.dump(self.model_pipeline.get_model(), f"models/{name}.joblib")
        return self

    def train(self):
        X = self.data.drop("Sponsored", axis=1)
        y = self.data["Sponsored"]
        self.fit_model(X,y)
        return self


if __name__ == "__main__":
    trainer = Trainer(SponsorClfXGBPipeline())
    trainer.train()
    trainer.export_model("sponsor_clf_01")