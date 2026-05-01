from fastapi import APIRouter
from sklearn.preprocessing import StandardScaler
from fastapi.responses import Response
from ml import utils
from ml import train
from typing import Dict
import pandas as pd
import joblib

router = APIRouter()

@router.get("/")
def running():
    return {"status":"Application running"}

@router.get("/test_model")
def test_model(station_id:str, days:int)->Dict[str,int]:
    df_merged = utils.get_station_df(station_id,days)

    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)

    df_merged_past_test_set_predictors, df_merged_past_test_set_labels = utils.extract_predictors_labels(df_merged_past_test_set)

    forest_reg = joblib.load("forest_reg.pkl")
    forest_reg_rmse = utils.get_forest_rmse(forest_reg, df_merged_past_test_set_predictors, df_merged_past_test_set_labels)
    return {"RMSE": forest_reg_rmse}

@router.get("/plot_test",response_class=Response)
def plot_test(station_id:str, days:int)->Response:
    scaler = StandardScaler()
    df_merged = utils.get_station_df(station_id,days)
    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
    df_merged_past_test_set_copy = df_merged_past_test_set.copy()
    numeric_cols = utils.extract_numeric_columns(df_merged_past_test_set_copy)
    df_merged_past_test_set_copy[numeric_cols] = scaler.fit_transform(df_merged_past_test_set_copy[numeric_cols])
    
    return utils.plot(df_merged_past_test_set_copy, "Past Test Set")
    #print(df_merged_past_test_set_copy)

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

    return utils.plot(df_merged_past_training_set_copy, "Past Training Set")

@router.get("/future_set")
def future_set(station_id:str):
    scaler = StandardScaler()

    df_merged = utils.get_station_df(station_id, days = 30)

    df_merged_future = utils.get_future_df(df_merged)

    df_merged_future_predictors, df_merged_future_labels = utils.extract_predictors_labels(df_merged_future)

    forest_reg = joblib.load("forest_reg.pkl")
    predictions = utils.test_model(forest_reg, df_merged_future_predictors)

    predictions = pd.DataFrame(predictions, columns=['levelAtHour'])
    predictions['measuredAt'] = df_merged_future['measuredAt']
       
    df_merged_future_predictions = pd.merge(df_merged_future.drop(columns = ['levelAtHour']), predictions[['measuredAt', 'levelAtHour']], on='measuredAt', how ='left')

    df_merged_future_predictions_copy = df_merged_future_predictions.copy()
    numeric_cols = utils.extract_numeric_columns(df_merged_future_predictions_copy)

    df_merged_future_predictions_copy[numeric_cols] = scaler.fit_transform(df_merged_future_predictions_copy[numeric_cols])

    return (df_merged_future_predictions_copy).to_dict(orient='records')

@router.get("/plot_future")
def plot_future(station_id:str)->Response:
    df_merged_future_predictions_copy = pd.DataFrame(future_set(station_id))
    return utils.plot(df_merged_future_predictions_copy, "Future Predictions")

