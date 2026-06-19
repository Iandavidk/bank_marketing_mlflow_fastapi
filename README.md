# Bank Marketing Campaign Prediction Portfolio Project

This repository demonstrates a complete machine-learning project for predicting whether a client will subscribe to a term deposit following a telemarketing campaign. It includes data processing, model training and selection, MLflow experiment tracking, a prediction API, an interactive Streamlit dashboard, and an asynchronous Kafka-based prediction pipeline for real-time ingestion.

The goal: present reproducible, production-oriented code and documentation suitable for a portfolio or demo.

---

**Contents**
- `src/` — application source: `app.py` (FastAPI), `train.py`, `streamlit_app.py`, `kafka_producer.py` and helpers.
- `models/` — trained model artifacts and metadata (`bank_marketing_model.joblib`, `best_model.json`).
- `notebooks/` — exploratory and reporting Jupyter notebooks.
- `reports/` — exported reports and artifacts (HTML/PDF/CSVs).
- `data/` — raw or sample data used for experiments (not all datasets included due to size).
- `mlruns/` — MLflow experiment runs (if you run experiments locally).

---

## Project overview

1. Data: the UCI Bank Marketing dataset (via OpenML). The dataset contains client demographics and campaign call attributes.
2. Preprocessing: numeric imputation + scaling, categorical imputation + one-hot encoding using `ColumnTransformer` inside scikit-learn `Pipeline`.
3. Models: baseline models trained and compared include Logistic Regression, Random Forest, and Gradient Boosting. The best model pipeline is saved to `models/`.
4. Tracking: MLflow logs parameters, metrics, and model artifacts for each run.
5. Serving: a FastAPI service (`src.app`) exposes a REST endpoint `/predict` and starts a background Kafka consumer that reads input messages, performs predictions, and publishes results asynchronously.
6. Frontend: a Streamlit dashboard (`src/streamlit_app.py`) provides an interactive UI for manual predictions.
7. Streaming: `src/kafka_producer.py` demonstrates publishing CSV rows to Kafka to be consumed by the API at its own pace.

---

## Quickstart

Prerequisites

- Python 3.9+ virtual environment
- Docker (recommended) for Kafka and Zookeeper or access to a Kafka cluster

Create virtual environment and install dependencies:

```bash
python -m venv .env
.env\Scripts\activate    # Windows
source .env/bin/activate # macOS / Linux
pip install -r requirements.txt
```

Train a model (optional — pre-trained model is included):

```bash
mlflow ui  # optional: view runs at http://127.0.0.1:5000
python src/train.py
```

Run the FastAPI service:

```bash
uvicorn src.app:app --reload
```

Open the Streamlit dashboard:

```bash
streamlit run src/streamlit_app.py
```

---

## Kafka asynchronous prediction pipeline

This project includes a Kafka-based ingestion flow so you can stream many input records asynchronously and let the prediction API consume them at its own pace.

1. Start Kafka & Zookeeper (example using Docker):

```bash
docker run -d --name zookeeper -p 2181:2181 zookeeper:3.7
docker run -d --name kafka --link zookeeper -p 9092:9092 -e KAFKA_ZOOKEEPER_CONNECT=host.docker.internal:2181 -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 bitnami/kafka:latest
```

2. Start the API (it will start a background Kafka consumer on startup if `kafka-python` is installed):

```bash
uvicorn src.app:app --reload
```

3. Publish CSV rows to Kafka using the provided producer script:

```bash
python src/kafka_producer.py --csv data/inputs.csv --bootstrap localhost:9092 --topic predictions-input --delay 0.5
```

4. The API consumes messages from `predictions-input`, computes predictions, and publishes results to `predictions-output`.

Notes
- The Kafka consumer runs only if `kafka-python` is installed and Kafka is accessible. The app prints a warning if Kafka integration is disabled.
- The sample `kafka_producer.py` sends each CSV row as a JSON-encoded message. Adjust the CSV and schemas to match model input expectations.

---

## API specification

Endpoints (FastAPI)

- `GET /` — health/info
- `GET /health` — basic health check
- `POST /predict` — synchronous prediction for a single client payload (JSON). Example payload:

```json
{
  "age": 35,
  "job": "technician",
  "marital": "single",
  "education": "tertiary",
  "default": "no",
  "balance": 1200,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 12,
  "month": "may",
  "duration": 180,
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

Response:

```json
{
  "prediction": "yes",
  "subscription_probability": 0.8421
}
```

The background Kafka consumer publishes messages with schema:

```json
{
  "input": { ... original input ... },
  "prediction": "yes|no",
  "subscription_probability": 0.8421
}
```

---

## Model & Training details

- The training pipeline uses scikit-learn `Pipeline` combining `ColumnTransformer`:
  - Numeric: `SimpleImputer(median)` + `StandardScaler()`
  - Categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`
- Candidate models: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`.
- Evaluation: accuracy, precision, recall, f1, and ROC AUC. The selected model is saved to `models/bank_marketing_model.joblib`.

---

## Project structure (detailed)

- `src/` — source code
  - `app.py` — FastAPI app + optional Kafka consumer
  - `train.py` — training script that logs to MLflow and saves best model to `models/`
  - `streamlit_app.py` — interactive dashboard
  - `kafka_producer.py` — example producer to publish CSV rows
- `models/` — trained model artifacts
- `notebooks/` — exploratory notebooks and reports
- `reports/` — generated reports and evaluation CSVs
- `data/` — raw data (if present)
- `mlruns/` — MLflow tracking data (if you run experiments locally)

---

## Reproducibility & tips

- Use the included `requirements.txt` to reproduce the environment. Consider creating an environment with pinned package versions for exact reproducibility.
- For production deployment, consider:
  - Packaging the model and API in Docker containers
  - Using a managed Kafka cluster (Confluent, MSK, Aiven)
  - Adding authentication and rate-limiting to public endpoints
  - Persisting predictions to a durable store (database) and adding monitoring/alerting

---

## Troubleshooting

- If the API fails to start due to `kafka-python` missing, install it with `pip install kafka-python` or remove/disable Kafka integration.
- If model loading fails, ensure `models/bank_marketing_model.joblib` exists or run `python src/train.py`.

---

## License & Attribution

This project is provided for portfolio and educational purposes. Data originates from the UCI Bank Marketing dataset (via OpenML).
