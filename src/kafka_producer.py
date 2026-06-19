"""Simple Kafka producer to stream CSV rows as messages.

Usage:
  python src/kafka_producer.py --csv data/inputs.csv --bootstrap localhost:9092 --topic predictions-input --delay 0.5
"""
import argparse
import json
import time
import pandas as pd
from kafka import KafkaProducer


def send_csv_rows(csv_path: str, bootstrap: str, topic: str, delay: float = 0.5):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        payload = row.to_dict()
        producer.send(topic, payload)
        print(f"sent -> {payload}")
        time.sleep(delay)
    producer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file with input rows")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka bootstrap server")
    parser.add_argument("--topic", default="predictions-input", help="Kafka topic to publish to")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between messages (s)")
    args = parser.parse_args()
    send_csv_rows(args.csv, args.bootstrap, args.topic, args.delay)


if __name__ == "__main__":
    main()
