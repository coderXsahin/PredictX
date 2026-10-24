
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "UP"
    assert data["application"] == "PredictX"
    assert data["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 45.0,
        "tool_wear": 180,
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["status"] in ["HIGH_RISK", "LOWER_RISK"]
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["threshold"] == 0.4
    assert data["model"] == "Random Forest"


def test_invalid_machine_type_is_rejected():
    payload = {
        "machine_type": "X",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 45.0,
        "tool_wear": 180,
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


def test_invalid_sensor_value_is_rejected():
    payload = {
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": -10,
        "torque": 45.0,
        "tool_wear": 180,
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422

def test_prediction_is_saved_and_appears_in_history():
    payload = {
        "machine_type": "M",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 45.0,
        "tool_wear": 180,
    }

    prediction_response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert prediction_response.status_code == 200
    prediction = prediction_response.json()

    assert isinstance(prediction["id"], int)
    assert prediction["id"] > 0
    assert prediction["created_at"]
    assert 0.0 <= prediction["failure_probability"] <= 1.0

    history_response = client.get("/api/v1/predictions?limit=100")

    assert history_response.status_code == 200
    history = history_response.json()

    matching_records = [
        item
        for item in history["items"]
        if item["id"] == prediction["id"]
    ]

    assert history["count"] >= 1
    assert len(matching_records) == 1
    assert matching_records[0]["machine_type"] == "M"
