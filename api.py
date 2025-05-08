from pydantic import BaseModel
import torch
import dill
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pydantic import BaseModel
from typing import List

from classes import SimpleAE, SimpleTensorDataset
from fastapi import FastAPI

app = FastAPI()


class Form(BaseModel):
    records: List[dict]

class Prediction(BaseModel):
    prediction: object


with open('default_prediction_model.pkl', 'rb') as file:
    model = dill.load(file)


@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame(form.records)

    preprocessed_df = model['model'].preprocessor.transform(df)
    y_pred = model['model'].model.predict_proba(preprocessed_df)[:, 1]

    return {"prediction": y_pred.tolist()}

