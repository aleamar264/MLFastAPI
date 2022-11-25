import pandas as pd
from ..ms import model

def predict (x, model):
    return model.predict(x)[0]

def get_model_response(input) -> dict:
    X = pd.json_normalize(input.__dict__)
    prediction = predict(X, model)
    if prediction == 1:
        label = "M"
    else:
        label = "B"
    return {
        "label" : label,
        "prediction" : int(prediction)
    }