
import os
import joblib

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
ml_models ={}



class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float



def load_model(path: str):
    model = None
    with open(path, "rb") as f:
        model = joblib.load(f)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
 # Load the ML model
    ml_models["logistic_model"] = load_model(os.getenv("LOGISTIC_MODEL"))
    ml_models["rf_model"] = load_model(os.getenv("RF_MODEL"))


    yield
    # This code will be executed after the application finishes handling requests, right before the shutdown
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Models loaded and FastAPI is ready!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    # Return the list of available models' names
    return {"available_models": list(ml_models.keys())}


@app.post("/predict/{model_name}")
async def predict(model_name: str, iris: IrisData):
    input_data = [
        [iris.sepal_length, 
         iris.sepal_width, 
         iris.petal_length, 
         iris.petal_width]
         ]

    # input_data = [[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]]
    
    if model_name not in ml_models.keys():
        raise HTTPException(status_code=404, detail="Model not found")
    
    ml_model = ml_models[model_name]
    prediction = ml_model.predict(input_data)
    prediction_proba = ml_model.predict_proba(input_data)

    return {
        "model": model_name, 
        "prediction": int(prediction[0]),
        "probabilities": prediction_proba[0].tolist()
        }





print("Successfull!")