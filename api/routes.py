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
import requests
import os
import numpy as np

class ModelRequest(BaseModel):
    station_id: str
    days: Optional[int] = 50
    level: Optional[int] = 3.0

router = APIRouter()

@router.get("/")
def running()->dict[str,list[str]]:
    return {"status": os.listdir("models")}

@router.get("/train_model")
def train_model(request: ModelRequest = Depends()) -> dict[str, str]:
    print("Training model")
    print("DF merged")

    df_merged = utils.get_station_df(request.station_id,request.days)
    
    print("Df merged past training and test set")
    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
    
    print("past training set labels & predictors")
    df_merged_past_training_set_predictors, df_merged_past_training_set_labels = utils.extract_predictors_labels(df_merged_past_training_set)

    print("forest_reg model")
    forest_reg = train.create_forest(
            df_merged_past_training_set_predictors, 
            df_merged_past_training_set_labels
    )

    print("Model Path")
    model_path = f"models/forest_reg_{request.station_id}.pkl"

    print("Dumping model")
    joblib.dump(forest_reg, model_path)
    print("Returning status")
    return {'status':"Finished Re-training The Model"}

@router.get("/test_model")
def test_model(request: ModelRequest = Depends())->dict[str,float]:
    print("Load station data")
    df_merged = utils.get_station_df(request.station_id,request.days)
    print("Split train test")
    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
    print("Extract test features")
    df_merged_past_test_set_predictors, df_merged_past_test_set_labels = utils.extract_predictors_labels(df_merged_past_test_set)
    print("Build model path")
    model_path = f"models/forest_reg_{request.station_id}.pkl"
    print("Load trained model")
    forest_reg = joblib.load(model_path)
    print("Calculate model RMSE")
    #forest_reg = joblib.load("forest_reg.pkl")
    forest_reg_rmse = utils.get_forest_rmse(forest_reg, df_merged_past_test_set_predictors, df_merged_past_test_set_labels)
    print("Return RMSE result")
    return {"RMSE": forest_reg_rmse}

@router.get("/distribution")
def get_distribution(request: ModelRequest = Depends()):
    try:
        response = requests.get(f"https://api.weather.gc.ca/collections/hydrometric-daily-mean/items?STATION_NUMBER={request.station_id}&f=json&limit=10000&filter=properties.LEVEL IS NOT NULL")
        response.raise_for_status()
        # response.status_code is 200 at this point
        response = response.json()

        level = response.level
        rows = response['features']
        levels = []
        for row in rows:
            levels.append(row['properties']['LEVEL'])

        levels.sort()

        percentiles = list(range(0, 100, 5))

        percentile = np.interp(level, levels, percentiles)
        
        return percentile
    
    except requests.exceptions.Timeout:
        print("Request timeout")
    except requests.exceptions.RequestException as e:
        print("Request failed: ", e)
    except Exception as e:
        print("Failed: ",e)
    
    

    


@router.get("/plot_test",response_class=Response)
def plot_test(request: ModelRequest = Depends())->Response:
    #scaler = StandardScaler()
    df_merged = utils.get_station_df(request.station_id,request.days)
    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
    print(df_merged_past_test_set)
    print(df_merged_past_training_set)

    
    #df_merged_past_test_set_copy = df_merged_past_test_set.copy()

    #numeric_cols = utils.extract_numeric_columns(df_merged_past_test_set_copy)

    #print(df_merged_past_test_set_copy[numeric_cols])

    #df_merged_past_test_set_copy[numeric_cols] = scaler.fit_transform(df_merged_past_test_set_copy[numeric_cols])
    
    #return utils.plot(df_merged_past_test_set_copy, "Past Test Set")
    #print(df_merged_past_test_set_copy)
    return utils.plot(df_merged_past_test_set)





@router.get("/plot_train",response_class=Response)
def plot_train(request: ModelRequest = Depends())->Response:
    print("Plot Train")
    #scaler = StandardScaler()

    df_merged = utils.get_station_df(request.station_id,request.days)

    print(df_merged)

    df_merged_past_training_set, df_merged_past_test_set = utils.get_past_training_test_df(df_merged)
        
    #df_merged_past_training_set_copy = df_merged_past_training_set.copy()

    #numeric_cols = utils.extract_numeric_columns(df_merged_past_training_set_copy)

    #df_merged_past_training_set_copy[numeric_cols] = scaler.fit_transform(df_merged_past_training_set_copy[numeric_cols])

    return utils.plot(df_merged_past_training_set)


@router.get("/plot_future")
def plot_future(request: ModelRequest = Depends())->Response:
    df_merged_future_predictions_copy = pd.DataFrame(future_set(request))

    return utils.plot(df_merged_future_predictions_copy, 'future')

@router.get("/future_set")
def future_set(request: ModelRequest = Depends()):
    df_merged = utils.get_station_df(request.station_id, request.days)

    df_merged_past, df_merged_future = utils.get_future_df(df_merged)

    df_merged_future = pd.concat(
        [
            df_merged_past.iloc[-30:],
            df_merged_future
        ],
        ignore_index=True
    )

    df_merged_future_predictors, df_merged_future_labels = utils.extract_predictors_labels(df_merged_future)

    model_path = f"models/forest_reg_{request.station_id}.pkl"
    forest_reg = joblib.load(model_path)

    predictions = utils.test_model(forest_reg, df_merged_future_predictors)

    predictions = pd.DataFrame(
        predictions,
        columns=["levelAtHour"]
    ).reset_index(drop=True)

    predictions["measuredAt"] = (
        df_merged_future["measuredAt"]
        .reset_index(drop=True)
    )

    df_merged_future_predictions = (
        df_merged_future
            .drop(columns=["levelAtHour"])
            .merge(
                predictions,
                on="measuredAt",
                how="left"
            )
    )

    return df_merged_future_predictions.to_dict(orient="records")

