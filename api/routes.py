from fastapi import APIRouter
from sklearn.preprocessing import StandardScaler
from fastapi.responses import Response
from ml import utils
from ml import train
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import Optional
from fastapi import Depends

class ModelRequest(BaseModel):
    station_id: str
    days: Optional[int] = 100

router = APIRouter()

@router.get("/")
def running():
    return {"status":"Application running"}


@router.get("/train_model")
def train_model(request: ModelRequest = Depends()) -> dict[str, str]:
    df_merged = utils.get_station_df(request.station_id,request.days)

    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)

    df_merged_past_training_set_predictors, df_merged_past_training_set_labels = utils.extract_predictors_labels(df_merged_past_training_set)

    forest_reg = train.create_forest(
            df_merged_past_training_set_predictors, 
            df_merged_past_training_set_labels
    )

    model_path = f"models/forest_reg_{request.station_id}.pkl"

    joblib.dump(forest_reg, model_path)

    return {'status':"Finished Re-training The Model"}

@router.get("/test_model")
def test_model(request: ModelRequest = Depends())->dict[str,float]:
    df_merged = utils.get_station_df(request.station_id,request.days)

    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)

    df_merged_past_test_set_predictors, df_merged_past_test_set_labels = utils.extract_predictors_labels(df_merged_past_test_set)

    model_path = f"models/forest_reg_{request.station_id}.pkl"

    forest_reg = joblib.load(model_path)

    #forest_reg = joblib.load("forest_reg.pkl")

    forest_reg_rmse = utils.get_forest_rmse(forest_reg, df_merged_past_test_set_predictors, df_merged_past_test_set_labels)
    return {"RMSE": forest_reg_rmse}



@router.get("/plot_test",response_class=Response)
def plot_test(request: ModelRequest = Depends())->Response:
    scaler = StandardScaler()
    df_merged = utils.get_station_df(request.station_id,request.days)
    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
    df_merged_past_test_set_copy = df_merged_past_test_set.copy()
    numeric_cols = utils.extract_numeric_columns(df_merged_past_test_set_copy)
    df_merged_past_test_set_copy[numeric_cols] = scaler.fit_transform(df_merged_past_test_set_copy[numeric_cols])
    
    return utils.plot(df_merged_past_test_set_copy, "Past Test Set")
    #print(df_merged_past_test_set_copy)



@router.get("/plot_train",response_class=Response)
def plot_train(request: ModelRequest = Depends())->Response:
    scaler = StandardScaler()

    df_merged = utils.get_station_df(request.station_id,request.days)

    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
    
    df_merged_past_training_set_copy = df_merged_past_training_set.copy()

    numeric_cols = utils.extract_numeric_columns(df_merged_past_training_set_copy)

    df_merged_past_training_set_copy[numeric_cols] = scaler.fit_transform(df_merged_past_training_set_copy[numeric_cols])

    return utils.plot(df_merged_past_training_set_copy, "Past Training Set")

@router.get("/future_set")
def future_set(request: ModelRequest = Depends()):
    scaler = StandardScaler()

    df_merged = utils.get_station_df(request.station_id, request.days)

    df_merged_future = utils.get_future_df(df_merged)

    df_merged_future_predictors, df_merged_future_labels = utils.extract_predictors_labels(df_merged_future)

    model_path = f"models/forest_reg_{request.station_id}.pkl"

    forest_reg = joblib.load(model_path)

    predictions = utils.test_model(forest_reg, df_merged_future_predictors)

    predictions = pd.DataFrame(predictions, columns=['levelAtHour'])
    predictions['measuredAt'] = df_merged_future['measuredAt']
       
    df_merged_future_predictions = pd.merge(df_merged_future.drop(columns = ['levelAtHour']), predictions[['measuredAt', 'levelAtHour']], on='measuredAt', how ='left')

    df_merged_future_predictions_copy = df_merged_future_predictions.copy()
    numeric_cols = utils.extract_numeric_columns(df_merged_future_predictions_copy)

    df_merged_future_predictions_copy[numeric_cols] = scaler.fit_transform(df_merged_future_predictions_copy[numeric_cols])

    return (df_merged_future_predictions_copy).to_dict(orient='records')

@router.get("/plot_future")
def plot_future(request: ModelRequest = Depends())->Response:
    df_merged_future_predictions_copy = pd.DataFrame(future_set(request))
    return utils.plot(df_merged_future_predictions_copy, "Future Predictions")

