from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time
import logging
import json
import socket
import logging
import os

from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)



class LogstashHandler(logging.Handler):
    def emit(self, record):
        try:
            event = {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "model_version": "1.0.0",
                "application": "ml-churn-api",
            }

            if record.exc_info:
                import traceback
                event["error"] = "".join(
                    traceback.format_exception(*record.exc_info)
                )

            payload = (
                json.dumps(event) + "\n"
            ).encode("utf-8")

            host = os.getenv("LOGSTASH_HOST", "logstash")
            port = int(os.getenv("LOGSTASH_PORT", "5000"))

            with socket.create_connection(
                (host, port),
                timeout=0.5
            ) as connection:
                connection.sendall(payload)

        except (OSError, ValueError):
            # Logstash indisponible :
            # la prédiction continue normalement.
            pass


logger.addHandler(LogstashHandler())



app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0"
)

model = joblib.load("models/model.pkl")


class ClientData(BaseModel):
    age: int
    anciennete: int
    revenu: float
    nb_reclamations: int
    montant_mensuel: float
    retards_paiement: int


@app.get("/")
def root():
    return {
        "application": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "UP"
    }


@app.get("/model/info")
def model_info():
    return {
        "model": "Random Forest",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict(data: ClientData):

    start = time.time()
    logger.info("Prediction request: model_version=1.0.0")

    try:
        values = [[
            data.age,
            data.anciennete,
            data.revenu,
            data.nb_reclamations,
            data.montant_mensuel,
            data.retards_paiement
        ]]

        prediction = model.predict(values)[0]
        probability = model.predict_proba(values)[0][1]

        duration = time.time() - start

        result = "CHURN" if prediction == 1 else "NO_CHURN"

        logger.info(
            "Prediction completed: model_version=1.0.0 "
            "result=%s probability=%.4f duration_ms=%.2f",
            result,
            float(probability),
            duration * 1000
        )

        return {
            "prediction": result,
            "probability": round(float(probability), 4),
            "model_version": "1.0.0",
            "duration_ms": round(duration * 1000, 2)
        }
    except Exception:
        logger.error("Prediction error")
        raise

Instrumentator().instrument(app).expose(app)