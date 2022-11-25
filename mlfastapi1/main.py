from .ms import app
from .ms.function import get_model_response
from pydantic import BaseModel, Field

model_name = "Breast Cancer Winsconsin (Diagnostic)"
version = "v1.0.0"

# Input data for validation
class Input(BaseModel):
    concavity_mean: float = Field(..., gt=0)
    concave_points_mean: float = Field(..., gt=0)
    perimeter_se: float = Field(..., gt=0)
    area_se: float = Field(..., gt=0)
    texture_worst: float = Field(..., gt=0)
    area_worst: float = Field(..., gt=0)

    class Config:
            schema_extra = {
            "concavity_mean": 0.3001,
            "concave_points_mean": 0.1471,
            "perimeter_se": 8.589,
            "area_se": 153.4,
            "texture_worst": 17.33,
            "area_worst": 2019.0,
        }

class Output(BaseModel):
    label : str
    prediction : int

@app.get('/info')
async def model_info():
    """ Return model information, version, how to call """
    return {
        "name" : model_name,
        "version" : version
    }

@app.get('/health')
async def service_health():
    """ Return service health """
    return {
        "ok"
    }

@app.post('/predict', response_model = Output)
async def model_predict(input: Input):
    """ Predict with input """
    return get_model_response(input)