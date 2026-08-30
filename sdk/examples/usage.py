import pandas as pd
import numpy as np
from driftwatch import DriftClient

client = DriftClient(api_url="http://127.0.0.1:8000")

np.random.seed(42)
training_data = pd.DataFrame(
    {
        "age": np.random.normal(35, 10, 500).astype(int),
        "income": np.random.normal(60000, 15000, 500),
        "region": np.random.choice(["north", "south", "east", "west"], 500),
    }
)

production_data = pd.DataFrame(
    {
        "age": np.random.normal(55, 8, 200).astype(int),
        "income": np.random.normal(120000, 20000, 200),
        "region": np.random.choice(["west", "west", "east"], 200),
    }
)

print("Registering model...")
model = client.register_model(
    name="fraud-detector-sdk-test",
    feature_names=["age", "income", "region"],
    model_type="classification",
    description="Registered via DriftWatch SDK",
)
print(f"Model registered: {model.id}")

print("Uploading baseline...")
client.log_baseline(
    model_id=model.id,
    dataframe=training_data,
    categorical_features=["region"],
)
print("Baseline uploaded.")

print("Submitting production batch...")
report = client.log(
    model_id=model.id,
    dataframe=production_data,
    categorical_features=["region"],
)

print(f"Status: {report['status']}")
print(f"Overall drift score: {report['overall_drift_score']}")
print(f"Drifted features: {report['drifted_features']}")
