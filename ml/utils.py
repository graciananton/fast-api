import requests
from dotenv import load_dotenv
from openai import OpenAI
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
from datetime import datetime, UTC
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
import re
import json
from pathlib import Path

current_working_script_dir = Path(__file__).resolve().parent
env_dir = current_working_script_dir.parent

load_dotenv(dotenv_path = env_dir / ".env")

open_ai_api_key = os.getenv("OPENAI_API_KEY")
print("open_ai_api_key")
print(open_ai_api_key)

def get_stations()->dict:
    res = requests.get("https://gracian.ca/laravel/public/api/stations")
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

def generate_station_message(station_id):
    print("open ai key")
    print(open_ai_api_key)
    client = OpenAI(
        api_key=open_ai_api_key
    )

    stats = requests.get("http://gracian.ca/laravel/public/api/stats?stationId="+station_id)

    predictions = requests.get('http://gracian.ca/laravel/public/api/future?stationId='+station_id+'&order=desc&limit=48')

    predictions = predictions.json()
    stats = stats.json()

    df_predictions = pd.DataFrame(predictions)

    print(stats)
    print(type(stats))

    print(df_predictions)

    df_predictions['percentile'] = df_predictions['percentile'].apply(float)

    mean_percentile = sum(df_predictions['percentile']) / len(df_predictions)

    stats['meanPercentile'] = mean_percentile

    response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        instructions="Write a concise summary of the data.",
        input=json.dumps(stats)
    )

    return {"message": response.output_text}


def plot(df, category = "past")->Response:
    

    if category == "future":
        fig, (predictions, weather) = plt.subplots(1, 2, figsize =(13,4))

        current_time = datetime.now(UTC).replace(
            minute=0,
            second=0,
            microsecond=0
        ).isoformat()
        x_text = (current_time + timedelta(minutes=45)).isoformat()
        x_border = (current_time - timedelta(minutes = 60)).isoformat()

        current_time = pd.to_datetime(current_time)
        x_text = pd.to_datetime(x_text)
        x_border = pd.to_datetime(x_border)

        before = df[df['measuredAt'] <= current_time]

        after = df[df['measuredAt'] > x_border]

        level = df.loc[df['measuredAt'] == current_time, 'levelAtHour'].iloc[0]
        
        # used to be ax -> predictions
        predictions = before.plot(
            kind = 'line', 
            x='measuredAt', 
            y='levelAtHour', 
            color='#0057E7',
            ax = predictions
        )

        ax.get_legend().remove()

        ax.spines["top"].set_linewidth(0)

        ax.spines["bottom"].set_color("gray")
        ax.spines["bottom"].set_linewidth(1)

        ax.spines["left"].set_color("gray")
        ax.spines["left"].set_linewidth(1)

        ax.spines["right"].set_linewidth(0)
        
        ax2 = after.plot(
            kind = 'line', 
            x='measuredAt', 
            y='levelAtHour', 
            color='#0057E7',
            linestyle="--", 
            ax = ax
        )

        timesAfter = map(getTimes, ax2.get_xticklabels())

        positions = ax2.get_xticks()
    
        ax2.set_xticks(positions)

        ax2.set_xticklabels(timesAfter, rotation=0, ha='center')


        ax.scatter(
            current_time,
            level,
            color = '#0057E7',
            marker = 'o',
            s=30
        )

        ax.axvline(
            x = current_time,
            color = 'gray',
            linewidth = 1,
            linestyle = "--"
        )

        ax.text(
            x = x_text,
            y = level + 0.0001,
            s = str(round(level,2)) + "m", # get 0th indexed value from series pandas
            fontsize=10,
            color="gray",
            fontweight = 'bold'
        )

        ax.set_xlabel("Time (Toronto/America)")
        ax.set_ylabel("Water Level (m)")
        ax2.get_legend().remove()

        ax3 = before[['precipitation','measuredAt']].plot(
            kind = "line",
            x = "measuredAt",
            y = "precipitation",
            color='#0057E7',
            ax = weather
        )

        after[['precipitation','measuredAt']].plot(
            kind = 'line',
            linestyle='--',
            x='measuredAt',
            y='precipitation',
            color='#0057E7',
            ax = ax3
        )

        ax3.set_xlabel("Time (Toronto/America)")
        ax3.set_ylabel("Precipitation (mm)")
        
    else:
        plt.figure()
        ax = df.plot(
            kind = 'line', 
            x='measuredAt', 
            y='levelAtHour', 
            color='#0057E7'
        )

        ax.spines["top"].set_linewidth(0)

        ax.spines["bottom"].set_color("gray")
        ax.spines["bottom"].set_linewidth(1)

        ax.spines["left"].set_color("gray")
        ax.spines["left"].set_linewidth(1)

        ax.spines["right"].set_linewidth(0)

        #timesAfter = map(getTimes, ax.get_xticklabels())

        #positions = ax.get_xticks()
    
        #ax.set_xticks(positions)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')

    buffer = io.BytesIO()

    plt.savefig(buffer, format="png")
    plt.close()

    return Response(content=buffer.getvalue(), media_type="image/png")

def getTimes(tickLabel):
    x = tickLabel.get_position()[0]

    dt = mdates.num2date(x, tz=ZoneInfo("America/Toronto"))
    hour = dt.hour

    print("Hour:")
    print(hour)
    return (str(int(hour) - 12) + " PM") if hour >= 12 else str(hour) + " AM"


def test_model(model, predictors):
    predictions = model.predict(predictors)
    return predictions

def get_future_df(df_merged):
    df_merged_past, df_merged_future = split_past_future(df_merged)

    print("df merged future")
    print(df_merged_future)
    print(type(df_merged_future))

    return add_index(df_merged_past), add_index(df_merged_future)


def split_past_future(df):
    df_merged_past = df.iloc[:df['levelAtHour'].isna().idxmax() - 1]

    df_merged_future = df.iloc[df['levelAtHour'].isna().idxmax():len(df)]
    return df_merged_past, df_merged_future


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
    res = requests.get(url, timeout = 1200)
    print("Data result")
    print(res)

    data = res.json()
    return data
