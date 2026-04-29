from fastapi import APIRouter
from ml import utils
from sklearn.preprocessing import StandardScaler

router = APIRouter()

@router.get("/train_model")
def train_model():
    merged_df = utils.get_station_df("02KF005",12)
    numeric_cols = merged_df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()

    return merged_df.to_json()