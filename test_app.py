
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

valid_data = {
    "milk_quantity": 100,
    "collection_temp": 30,
    "transport_duration": 2,
    "cooling_delay": 0.5,
    "fat_percentage": 4,
    "acidity_indicator": 6.5,
    "container_type": "Steel",
    "ambient_temp": 28
}


def test_home():

    response = client.get("/")

    assert response.status_code == 200


def test_valid_prediction():

    response = client.post("/predict", json=valid_data)

    assert response.status_code == 200

    assert response.json()["prediction"] in [
        "Likely Accepted",
        "Rejection Risk"
    ]


def test_negative_quantity():

    data = valid_data.copy()

    data["milk_quantity"] = -10

    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_impossible_temperature():

    data = valid_data.copy()

    data["collection_temp"] = 150

    response = client.post("/predict", json=data)

    assert response.status_code == 422


def test_invalid_container():

    data = valid_data.copy()

    data["container_type"] = "Wood"

    response = client.post("/predict", json=data)

    assert response.status_code == 422
