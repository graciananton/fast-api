from fastapi import APIRouter
from ml import utils

router = APIRouter()

@router.get("/train_model")
def train_model():
    stations = utils.get_stations()
    return stations