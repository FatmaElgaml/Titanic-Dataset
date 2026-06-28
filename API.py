from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# for security (how websites show that and what http type and so on )
app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_headers =["*"],
    allow_methods =["*"],
    allow_credentials=True,
    )

import joblib
artifacts = joblib.load("titanic_model.pkl")

model = artifacts["model"]
scaler = artifacts["scaler"]

Sex_label = artifacts["Sex_label"]
Embarked_label = artifacts["Embarked_label"]
Title_label = artifacts["Title_label"]

from pydantic import BaseModel
class Passenger_input(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    Embarked: str
    Title: str


@app.get("/")
def home():
    return {"Titanic predection APi is working now."}

@app.post("/prediction")
def predict(data:Passenger_input):
    Sex = Sex_label.transform([data.Sex])[0]
    Embarked = Embarked_label.transform([data.Embarked])[0]
    Title = Title_label.transform([data.Title])[0]

    features =[[
        data.Pclass,
        Sex,
        data.Age,
        data.Fare,
        Embarked,
        Title
    ]]


    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    result = "Survived" if prediction == 1 else "Not Survived"

    return {
    "prediction": result
}