from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import json
import threading
import time
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

# Model artifact path (moved into repository `models/` folder)
MODEL_PATH = Path(__file__).parent.parent / "models" / "bank_marketing_model.joblib"
EXPERIMENT_NAME = 'bank_marketing_campaign_prediction'

app = FastAPI(title='Bank Marketing Prediction API', version='1.0.0')
model = joblib.load(MODEL_PATH)

# Kafka configuration (import kafka client if available)
try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except Exception:
    KafkaConsumer = None
    KafkaProducer = None
    KAFKA_AVAILABLE = False

KAFKA_BOOTSTRAP = "localhost:9092"
INPUT_TOPIC = "predictions-input"
OUTPUT_TOPIC = "predictions-output"


def _predict_and_publish(record: dict, producer):
    try:
        input_df = pd.DataFrame([record])
        pred_class = int(model.predict(input_df)[0])
        pred_prob = float(model.predict_proba(input_df)[:, 1][0])
        label_val = "yes" if pred_class == 1 else "no"
        result = {
            "input": record,
            "prediction": label_val,
            "subscription_probability": round(pred_prob, 4),
        }
        if KAFKA_AVAILABLE and producer is not None:
            producer.send(OUTPUT_TOPIC, value=result)
            producer.flush()
        producer.flush()
        return result
    except Exception as exc:
        return {"error": str(exc), "input": record}


def _kafka_consumer_loop():
    # Background consumer that reads input messages and publishes predictions
    try:
        consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
        )
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print("Kafka consumer started, listening for input messages...")
        while True:
            for msg in consumer:
                record = msg.value
                print(f"consumed -> {record}")
                res = _predict_and_publish(record, producer)
                print(f"published -> {res}")
            time.sleep(1)
    except Exception as exc:
        print(f"Kafka consumer loop error: {exc}")
mlflow.set_experiment(EXPERIMENT_NAME)

class BankClient(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get('/')
def home():
    return {'message': 'Bank Marketing Prediction API is running'}

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(client: BankClient):
    input_df = pd.DataFrame([client.dict()])
    pred_class = int(model.predict(input_df)[0])
    pred_prob = float(model.predict_proba(input_df)[:, 1][0])
    label_val = 'yes' if pred_class == 1 else 'no'
    with mlflow.start_run(nested=True):
        mlflow.log_params(client.dict())
        mlflow.log_metric('prediction_probability', pred_prob)
        mlflow.log_metric('prediction_class', pred_class)
    return {
        'prediction': label_val,
        'subscription_probability': round(pred_prob, 4)
    }


@app.on_event("startup")
def start_kafka_background_consumer():
    if not KAFKA_AVAILABLE:
        print("kafka-python not installed: Kafka consumer disabled. Install 'kafka-python' to enable.")
        return
    thread = threading.Thread(target=_kafka_consumer_loop, daemon=True)
    thread.start()