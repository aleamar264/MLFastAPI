from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load('mlfastapi1/model/model_binary_data.gz')