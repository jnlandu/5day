from fastapi.testclient import TestClient
from .main import app


client = TestClient(app)



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/health')
async def health():
    return {"health status": "healthy"}


def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"msg": "Healthy"}

def test_models():
    response = client.get("/models")
    assert response.status_code == 404
    assert response.json() == {"model": "log_reg"}

