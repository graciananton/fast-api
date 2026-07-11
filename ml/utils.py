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
    client = OpenAI(
        api_key=open_ai_api_key
    )

    stats = requests.get("http://gracian.ca/laravel/public/api/stats?stationId="+station_id)

    stats = stats.json()
    
    formatted_peak_time = convert_to_formatted_date(datetime.fromisoformat(stats['peakTime']))
    formatted_last_updated = convert_to_formatted_date(datetime.fromisoformat(stats['lastUpdated']))

    stats['peakTime'] = formatted_peak_time
    stats['lastUpdated'] = formatted_last_updated


    print(json.dumps(stats))
    
    response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        instructions="Write a concise summary of the data.",
        input=json.dumps(stats)
    )

    return {"message": response.output_text}

def convert_to_formatted_date(time):
    toronto_time = time.astimezone(ZoneInfo("America/Toronto"))

    return toronto_time.isoformat()

def plot(df, category = "past")->Response:
    

    if category == "future":
        fig, ((predictions, precipitation), (temperature, wind_speed)) = plt.subplots(2, 2, figsize =(13,6))

        current_time = datetime.now(UTC).replace(
            minute=0,
            second=0,
            microsecond=0
        )

        x_text = (current_time + timedelta(minutes=45)).isoformat()
        x_border = (current_time - timedelta(minutes = 60)).isoformat()

        current_time = current_time.isoformat()

        current_time = pd.to_datetime(current_time)
        x_text = pd.to_datetime(x_text)
        x_border = pd.to_datetime(x_border)

        before = df[df['measuredAt'] <= current_time]

        # after contains instances before the current time, therefore it is good to add the vertical line and indicator
        after = df[df['measuredAt'] > x_border]

        level = df.loc[df['measuredAt'] == current_time, 'levelAtHour'].iloc[0]
        precip = df.loc[df['measuredAt'] == current_time, 'precipitation'].iloc[0]
        temp = df.loc[df['measuredAt'] == current_time, 'temperature_2m'].iloc[0]
        wind = df.loc[df['measuredAt'] == current_time, 'wind_speed_10m'].iloc[0]

        # up to this point, we have been using utc curren time for all operations

        # used to be ax -> before_predictions
        before_predictions = before.plot(
            kind = 'line', 
            x='measuredAt', 
            y='levelAtHour', 
            color='#0057E7',
            ax = predictions
        )

        # used to be ax2 -> after_predictions
        after_predictions = after.plot(
            kind = 'line', 
            x='measuredAt', 
            y='levelAtHour', 
            color='#0057E7',
            linestyle="--", 
            ax = before_predictions
        )


        before_precipitation = before[['precipitation','measuredAt']].plot(
            kind = "line",
            x = "measuredAt",
            y = "precipitation",
            color='#17becf',
            ax = precipitation
        )

        after_precipitation = after[['precipitation','measuredAt']].plot(
            kind = 'line',
            linestyle='--',
            x='measuredAt',
            y='precipitation',
            color='#17becf',
            ax = before_precipitation
        )

        before_temperature = before[['temperature_2m','measuredAt']].plot(
            kind = "line",
            x = "measuredAt",
            y = "temperature_2m",
            color='#ff7f0e',
            ax = temperature
        )

        after_temperature = after[['temperature_2m','measuredAt']].plot(
            kind = 'line',
            linestyle='--',
            x='measuredAt',
            y='temperature_2m',
            color='#ff7f0e',
            ax = before_temperature
        )

        before_wind_speed = before[['wind_speed_10m','measuredAt']].plot(
            kind = "line",
            x = "measuredAt",
            y = "wind_speed_10m",
            color='#2ca02c',
            ax = wind_speed
        )

        after_wind_speed = after[['wind_speed_10m','measuredAt']].plot(
            kind = 'line',
            linestyle='--',
            x='measuredAt',
            y='wind_speed_10m',
            color='#2ca02c',
            ax = before_wind_speed
        )


        
        for ax in (predictions, precipitation, temperature, wind_speed):
            ax.get_legend().remove()
            ax.spines["top"].set_linewidth(0)
            ax.spines["bottom"].set_color("gray")
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["left"].set_color("gray")
            ax.spines["left"].set_linewidth(1)
            ax.spines["right"].set_linewidth(0)
            ax.set_xlabel("Time (Toronto/America)")

        after_predictions.set_ylabel("Water Level (m)")
        after_precipitation.set_ylabel("Precipitation (mm)")
        after_temperature.set_ylabel("Temperature (C)")
        after_wind_speed.set_ylabel("Wind Speed (C)")

        after_predictions.scatter(
            current_time,
            level,
            color = '#0057E7',
            marker = 'o',
            s=30
        )
        after_precipitation.scatter(
            current_time,
            precip,
            color = "#0057E7",
            marker = 'o',
            s = 30
        )
        after_temperature.scatter(
            current_time,
            temp,
            color = "#0057E7",
            marker = 'o',
            s = 30
        )
        after_wind_speed.scatter(
            current_time,
            wind,
            color = "#0057E7",
            marker = 'o',
            s = 30
        )


        after_predictions.text(
            x = x_text,
            y = level + 0.001,
            alpha = 1.0,
            s = str(round(level,2)) + "m", # get 0th indexed value from series pandas
            fontsize=10,
            color="gray",
            fontweight = 'bold'
        )
        
        after_precipitation.text(
            x = x_text,
            y = precip + 0.001,
            alpha = 1.0,
            s = str(round(precip,2)) + "mm", # get 0th indexed value from series pandas
            fontsize=10,
            color="gray",
            fontweight = 'bold'
        )
        
        after_temperature.text(
            x = x_text,
            y = temp + 0.001,
            alpha = 1.0,
            s = str(round(temp,2)) + "C", # get 0th indexed value from series pandas
            fontsize=10,
            color="gray",
            fontweight = 'bold'
        )

        after_wind_speed.text(
            x = x_text,
            y = wind + 0.001,
            alpha = 1.0,
            s = str(round(wind,2)) + "km/h", # get 0th indexed value from series pandas
            fontsize=10,
            color="gray",
            fontweight = 'bold'
        )



        for ax in (after_predictions, after_precipitation, after_temperature, after_wind_speed):
            ax.axvline(
                x = current_time,
                color = 'gray',
                linewidth = 1,
                linestyle = "--"
            )

            timesAfter = map(getTimes, ax.get_xticklabels())

            positions = ax.get_xticks()
        
            ax.set_xticks(positions)

            ax.set_xticklabels(timesAfter, rotation=0, ha='center')

        predictions.set_title("Predictions")
        precipitation.set_title("Precipitation")
        temperature.set_title("Temperature")
        wind_speed.set_title("Wind Speed")

        fig.subplots_adjust(
            hspace=0.5
        )
        # dot for current level/current rain text
        # current level/current rain text
        # ylabel
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
