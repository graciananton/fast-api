from fastapi import APIRouter
from ml import utils
from sklearn.preprocessing import StandardScaler
import joblib
from ml import train
from typing import Dict
from fastapi.responses import Response

router = APIRouter()

@router.get("/")
def running():
    return {"status":"Application running"}

@router.get("/train_model",response_class=Dict[str,str])
def train_model(station_id:str, days:int)->Dict[str,str]:
    df_merged = utils.get_station_df(station_id,days)

    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)

    df_merged_past_training_set_predictors, df_merged_past_training_set_labels = utils.extract_predictors_labels(df_merged_past_training_set)

    forest_reg = train.create_forest(
            df_merged_past_training_set_predictors, 
            df_merged_past_training_set_labels
    )
    joblib.dump(forest_reg, "forest_reg.pkl")

    return {'status':"Finished Re-training The Model"}

@router.get("/plot_train",response_class=Response)
def plot_train(station_id:str, days:int)->Response:
    scaler = StandardScaler()

    df_merged = utils.get_station_df(station_id,days)

    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
    
    df_merged_past_training_set_copy = df_merged_past_training_set.copy()

    numeric_cols = utils.extract_numeric_columns(df_merged_past_training_set_copy)

    df_merged_past_training_set_copy[numeric_cols] = scaler.fit_transform(df_merged_past_training_set_copy[numeric_cols])

    print(df_merged_past_training_set_copy.to_json())
    return utils.plot(df_merged_past_training_set_copy, "Past Training Set")


