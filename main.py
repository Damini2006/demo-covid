import joblib
import pandas as pd
from fastapi import FastAPI # pyright: ignore[reportMissingImports]
from pydantic import BaseModel # pyright: ignore[reportMissingImports]

data = joblib.load("D:\PROJECTS\demo-covid\ml_source\covid_diag.pkl")


class inp_data(BaseModel):
    Age:int
    Gender:int
    Fever:int
    Cough:int
    Fatigue:int
    Breathlessness:int
    Comordity:int
    Stage:int
    Type:int
    Tumor_Size:float
app=FastAPI()
@app.get("/")
def root_msg():
    return {"Message":"Welcome to karikalam magic show"}
@app.post("/predict")
def prediction(Data:inp_data):
    #inp=pd.DataFrame([Data.dict()])
    inp=np.array([[Data.Age,Data.Gender,Data.Fever,Data.Cough,Data.Fatigue,Data.Breathlessness,Data.Comordity,Data.Stage,Data.Type,Data.Tumor_Size]])
    prdd=data.predict(inp)[0]
    return {"prediction":prdd}