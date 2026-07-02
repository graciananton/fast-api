import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi.responses import Response
from fastapi.responses import FileResponse
import io
from sklearn.metrics import mean_squared_error
import numpy as np

load_dotenv() 

STATIONS_URL = os.getenv("STATIONS_URL")
READINGS_URL = os.getenv("READINGS_URL")
WEATHER_URL = os.getenv("WEATHER_URL")

def get_stations()->dict:
    res = requests.get(STATIONS_URL)
    stations = res.json()
    return stations

def get_days_ago(days:int):
    time_now = datetime.now(timezone.utc)
    time_ago = time_now - timedelta(days = days)
    formatted_time_ago = time_ago.strftime("%Y-%m-%d %H:%M:%S.%f")
    return formatted_time_ago

def get_station_df(station_id:str, days:int):

    days_ago = get_days_ago(days)

    readings_url = f"https://gracian.ca/laravel/public/api/readings?from={days_ago}&stationId={station_id}&f=json"

    readings_data = get_data_by_url(readings_url)

    weather_url = f"https://gracian.ca/laravel/public/api/weather?from={days_ago}&stationId={station_id}&f=json&limit=10000000000"
    weather_data = get_data_by_url(weather_url)

    weather_df = convert_to_df(weather_data)
    readings_df = convert_to_df(readings_data)

    weather_df['measuredAt'] = convert_to_datetime(weather_df['measuredAt'])
    readings_df['measuredAt'] = convert_to_datetime(readings_df['measuredAt'])

    readings_df = filter_valid_hours(readings_df)
    readings_df['level'] = convert_to_numeric(readings_df['level'])
    readings_df['measuredAtHour'] = readings_df['measuredAt'].dt.floor('h')
    readings_df['levelAtHour'] = readings_df.groupby('measuredAtHour')['level'].transform('mean')

    weather_expanded = weather_df['weather'].apply(pd.Series)
    
    weather_expanded_df = pd.concat(
        [weather_df[['measuredAt']], weather_expanded],
        axis=1
    )

    df_merged = create_merged_df(weather_expanded_df, readings_df)

    return df_merged

def extract_predictors_labels(df):
    df_predictors = df.drop(columns=['levelAtHour','measuredAt','stationId'])
    df_labels = df['levelAtHour']
    return df_predictors, df_labels

def extract_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    return numeric_cols

def plot(df, title)->Response:
    buf = io.BytesIO()
    #+" - "+df[['stationId']].iloc[0]['stationId']
    plt.figure()
    df.plot(
        kind = 'line', 
        x='measuredAt', 
        y='levelAtHour', 
        title = title
    )
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return Response(content=buf.getvalue(), media_type="image/png")

def test_model(model, predictors):
    predictions = model.predict(predictors)
    return predictions

def get_future_df(df_merged):
    df_merged_past, df_merged_future = split_past_future(df_merged)
    return add_index(df_merged_future)

def get_forest_rmse(forest_reg, predictors, labels):
    predictions = test_model(forest_reg, predictors)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return rmse

def get_past_training_test_df(df):
    df_merged_past, df_merged_future = split_past_future(df)

    df_merged_past_with_id = add_index(df_merged_past)
    df_merged_future_with_id = add_index(df_merged_future)

    df_merged_past_training_set, df_merged_past_test_set = split_train_test_by_id(df_merged_past_with_id,.2,'index')

    return df_merged_past_training_set, df_merged_past_test_set


def split_past_future(df):

    df_merged_past = df.loc[:df['levelAtHour'].isna().idxmax() - 1]

    df_merged_future = df.loc[df['levelAtHour'].isna().idxmax():len(df)]
    return df_merged_past, df_merged_future

def split_train_test_by_id(df_merged, test_ratio, id_column):
    ids = df_merged[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio * len(ids)))
    return df_merged.loc[~in_test_set], df_merged.loc[in_test_set]

def test_set_check(identifier, determining):
    return identifier < determining

def add_index(df):
    df = df.reset_index()
    return df


def create_merged_df(df_weather_expanded, df_readings):
    df_weather_expanded = df_weather_expanded[['measuredAt','temperature_2m','precipitation','snowfall','relative_humidity_2m','pressure_msl','rain','wind_speed_10m','stationId']].merge(
                    df_readings[['levelAtHour','measuredAtHour']], 
                    left_on = ['measuredAt'], 
                    right_on = ['measuredAtHour'], 
                    how = 'left'
                ).drop(columns=['measuredAtHour'])
    return df_weather_expanded

def convert_to_numeric(value):
    return pd.to_numeric(value, errors='coerce')

def filter_valid_hours(df)->pd.DataFrame:
    min_hour = get_min_hour(df, df['measuredAt'])
    max_hour = get_max_hour(df, df['measuredAt'])

    df = df[
        ~(
            (
                (df['measuredAt'].dt.floor('h') == min_hour) &
                (
                    (df['measuredAt'].dt.minute != 0) |
                    (df['measuredAt'].dt.second != 0)
                )
            ) |
            (
                (df['measuredAt'].dt.floor('h') == max_hour) &
                (
                    (df['measuredAt'].dt.minute != 0) |
                    (df['measuredAt'].dt.second != 0)
                )
            )
        )
    ]
    return df

def get_min_hour(df, value):
    return df['measuredAt'].dt.floor('h').min()
    
def get_max_hour(df, value):
    return df['measuredAt'].dt.floor('h').max()


def convert_to_datetime(value):
    return pd.to_datetime(value)

def convert_to_df(data):
    df = pd.DataFrame(data)
    return df

def get_data_by_url(url):
    res = requests.get(url)
    data = res.json()
    return data
