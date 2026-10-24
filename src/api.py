
from contextlib import asynccontextmanager
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.database import Base, SessionLocal, engine
from src.models import PredictionHistory

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "predictx_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run python src/train_model.py first."
    )

model = joblib.load(MODEL_PATH)
THRESHOLD = 0.40


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Importing PredictionHistory above registers its table with Base.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="PredictX API",
    description="ML-powered industrial machine failure prediction",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class MachineReading(BaseModel):
    machine_type: Literal["L", "M", "H"] = Field(
        description="Machine product type"
    )
    air_temperature: float = Field(ge=250, le=350)
    process_temperature: float = Field(ge=250, le=400)
    rotational_speed: int = Field(ge=1, le=10000)
    torque: float = Field(ge=0, le=1000)
    tool_wear: int = Field(ge=0, le=10000)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {
        "status": "UP",
        "application": "PredictX",
        "model_loaded": True,
    }


@app.post("/api/v1/predict")
def predict(
    reading: MachineReading,
    db: Session = Depends(get_db),
):
    features = pd.DataFrame([{
        "Type": reading.machine_type,
        "Air temperature [K]": reading.air_temperature,
        "Process temperature [K]": reading.process_temperature,
        "Rotational speed [rpm]": reading.rotational_speed,
        "Torque [Nm]": reading.torque,
        "Tool wear [min]": reading.tool_wear,
    }])

    probability = float(model.predict_proba(features)[0, 1])
    predicted_failure = probability >= THRESHOLD
    status = "HIGH_RISK" if predicted_failure else "LOWER_RISK"

    record = PredictionHistory(
        machine_type=reading.machine_type,
        air_temperature=reading.air_temperature,
        process_temperature=reading.process_temperature,
        rotational_speed=reading.rotational_speed,
        torque=reading.torque,
        tool_wear=reading.tool_wear,
        failure_probability=probability,
        threshold=THRESHOLD,
        prediction=int(predicted_failure),
        status=status,
    )

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Prediction was generated, but saving its history failed.",
        ) from exc

    return {
        "id": record.id,
        "prediction": record.prediction,
        "status": record.status,
        "failure_probability": round(record.failure_probability, 4),
        "threshold": record.threshold,
        "model": "Random Forest",
        "created_at": record.created_at.isoformat(),
        "message": (
            "Inspection recommended"
            if predicted_failure
            else "No high-risk alert at the configured threshold"
        ),
        "disclaimer": (
            "Experimental ML prediction, not a guarantee of machine safety."
        ),
    }


@app.get("/api/v1/predictions")
def prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    statement = (
        select(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .limit(limit)
    )
    records = db.scalars(statement).all()

    return {
        "count": len(records),
        "items": [
            {
                "id": record.id,
                "machine_type": record.machine_type,
                "air_temperature": record.air_temperature,
                "process_temperature": record.process_temperature,
                "rotational_speed": record.rotational_speed,
                "torque": record.torque,
                "tool_wear": record.tool_wear,
                "failure_probability": round(
                    record.failure_probability, 4
                ),
                "threshold": record.threshold,
                "prediction": record.prediction,
                "status": record.status,
                "created_at": record.created_at.isoformat(),
            }
            for record in records
        ],
    }
