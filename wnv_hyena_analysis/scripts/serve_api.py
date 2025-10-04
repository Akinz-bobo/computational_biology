#!/usr/bin/env python3
"""Minimal FastAPI service for on-demand classification.

Loads the best saved model artifacts (scaler, label encoder, model) from
`results/models/` and exposes a /classify endpoint that accepts raw sequences.
Falls back gracefully if deep model unavailable by using traditional features
extraction only.
"""

import json
from pathlib import Path
from typing import List, Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.analysis.data_loader import WNVDataLoader
from src.analysis.feature_extractor import TraditionalFeatureExtractor, HyenaDNAFeatureExtractor
from src.utils.logger import setup_logger


class SequencePayload(BaseModel):
    sequence_id: str = Field(..., description="Unique ID")
    sequence: str = Field(..., description="Raw nucleotide sequence (ACGTN)")


class ClassificationRequest(BaseModel):
    records: List[SequencePayload]
    return_features: bool = False


class ClassificationResponse(BaseModel):
    predictions: List[dict]
    model: str


def load_latest_model(models_dir: Path):
    # Heuristic: look for known model files and pick one (prefers xgboost -> rf -> svm)
    candidates = [
        ("xgboost_model.pkl", "XGBoost"),
        ("randomforest_model.pkl", "RandomForest"),
        ("svm_model.pkl", "SVM"),
        ("gradientboosting_model.pkl", "GradientBoosting"),
        ("logisticregression_model.pkl", "LogisticRegression"),
    ]
    for fname, name in candidates:
        path = models_dir / fname
        if path.exists():
            return joblib.load(path), name
    raise FileNotFoundError("No trained model artifacts found in results/models")


def build_feature_matrix(seqs: List[str], config_dict: dict):
    # Only traditional features for speed inside API – deep model optional
    extractor = TraditionalFeatureExtractor(config_dict)
    traditional = extractor.extract_features(seqs)
    if config_dict.get("features", {}).get("deep_learning", {}).get("enabled", True):
        try:
            deep = HyenaDNAFeatureExtractor(config_dict).extract_features(seqs)
            return np.hstack([traditional, deep])
        except Exception:
            return traditional
    return traditional


def create_app(config_path: str = "config/config.yaml") -> FastAPI:
    cfg = Config(config_path)
    logger = setup_logger("api", console=True)
    models_dir = Path(cfg.get("results.models_dir"))
    scaler_path = models_dir / "scaler.pkl"
    encoder_path = models_dir / "label_encoder.pkl"

    model, model_name = load_latest_model(models_dir)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    encoder = joblib.load(encoder_path) if encoder_path.exists() else None

    app = FastAPI(title="WNV Classification API", version="0.1.0")

    @app.get("/health")
    def health():  # pragma: no cover
        return {"status": "ok", "model": model_name}

    @app.post("/classify", response_model=ClassificationResponse)
    def classify(req: ClassificationRequest):
        if len(req.records) == 0:
            raise HTTPException(400, "No records provided")
        seqs = [r.sequence for r in req.records]
        feats = build_feature_matrix(seqs, cfg._config)
        if scaler is not None:
            feats = scaler.transform(feats)
        preds = model.predict(feats)
        probs = model.predict_proba(feats) if hasattr(model, "predict_proba") else None
        labels = encoder.inverse_transform(preds) if encoder is not None else preds
        out = []
        for i, r in enumerate(req.records):
            record = {"sequence_id": r.sequence_id, "prediction": str(labels[i])}
            if probs is not None:
                record["probabilities"] = probs[i].tolist()
            if req.return_features:
                record["features"] = feats[i].tolist()
            out.append(record)
        return ClassificationResponse(predictions=out, model=model_name)

    return app


app = create_app()  # Default application instance
