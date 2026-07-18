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

    current_hour = datetime.now().strftime("%Y-%m-%dT%H")

    url = f"http://gracian.ca/laravel/public/api/weather?stationId={station_id}&from={current_hour}&limit=1"

    current_weather = requests.get(url).json()[0]['weather']
    
    url = f"http://gracian.ca/laravel/public/api/stats?stationId={station_id}"

    stats = requests.get(url).json()

    current_level = stats['currentLevel']

    url = f"http://gracian.ca/laravel/public/api/weather?stationId={station_id}&from={current_hour}&limit=1&order=desc"

    projected_weather = requests.get(url).json()[0]['weather']

    url = f"http://gracian.ca/laravel/public/api/future?stationId={station_id}&order=desc&limit=1"

    projected_level = requests.get(url).json()[0]
    projected_level_prediction = projected_level['prediction']
    projected_level_status = projected_level['percentile']

    url = f"http://gracian.ca/laravel/public/api/stations?stationId={projected_level['stationId']}"
    station_name = requests.get(url).json()[0]['name']

    data = {
        "station_name": station_name,
        "current_weather":{
            "temperature": current_weather['temperature_2m'],
            "precipitation": current_weather['precipitation'],
            "wind_speed": current_weather['wind_speed_10m'],
        },
        "current_level":{
            "level": current_level
        },
        "forecast_weather":{
            "temperature": projected_weather['temperature_2m'],
            "precipitation": projected_weather['precipitation'],
            "wind_speed": projected_weather['wind_speed_10m']
        },
        "forecast_level":{
            "level": projected_level_prediction,
            'status': classify(float(projected_level_status))
        }
    }

    print(data)

    response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        instructions=
        "Summarize the following station data in a natural, concise paragraph. Focus on the key information only. Mention the station name, current weather, current water level, forecasted weather, and forecasted water level with its status. Avoid unnecessary wording, but don't make the summary feel abrupt.",
        input=json.dumps(data)
    )

    return {"message": response.output_text}

def classify(percentile):
    if percentile < 25:
        return "lower than usual"
    elif percentile < 65:
        return "normal"
    elif percentile < 80:
        return "above normal"
    else:
        return "extremely high"

def convert_to_formatted_date(time):
    toronto_time = time.astimezone(ZoneInfo("America/Toronto"))

    return toronto_time.isoformat()

def adjust_ticks(ax):
    timesAfter = map(getTimes, ax.get_xticklabels())

    positions = ax.get_xticks()

    ax.set_xticks(positions)

    ax.set_xticklabels(timesAfter, rotation=0, ha='center')

    return ax

def mapping(category, option):
    axes = {'temperature_2m':'Temperature (°C)', 'precipitation':'Precipitation (mm)', 'wind_speed_10m':'Wind Speed (km/hr)'}
    legend = {'temperature_2m':'Temperature', 'precipitation':'Precipitation', 'wind_speed_10m':'Wind Speed'}
    colors = {'temperature_2m':'#F59E0B', 'precipitation':'#D81B60', 'wind_speed_10m':'#10B981'}

    if category in axes:
        if option == 'axes':
            return axes[category]
        elif option == "colors":
            return colors[category]
        else:
            return legend[category]
    else:
        return axes['temperature_2m']


def customization(df, category, ax, ax2):

    ### ADJUST left, right, top, bottom SPINE COLORS ###
    ax.spines['left'].set_color("#3F76B8")
    ax2.spines['left'].set_color('#3F76B8')

    ax.spines['right'].set_color(mapping(category,"colors"))
    ax2.spines['right'].set_color(mapping(category,"colors"))
    
    ### ADJUST Y-TICK LABEL COLORS ###
    y_ticks = ax.yaxis.get_ticklabels()

    for y_tick in y_ticks:
        y_tick.set_color("#3F76B8")

    y_ticks = ax2.yaxis.get_ticklabels()
    
    for y_tick in y_ticks:
        y_tick.set_color(mapping(category,"colors"))


    ax.set_ylabel('Water Level (m)', color='#3F76B8')
    ax.set_xlabel("Measured At")

    ax2.set_ylabel(mapping(category, 'axes'), color = mapping(category,'colors'))

    ax.set_title(f"Water Level v. {mapping(category, 'legend')} - {df['stationId'].iloc[0]}")
    
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax2.grid(True, color="#E5E7EB", linewidth=0.8)    #ax2.spines['top'].set_visible(False)
    
    ax.get_legend()
    legend = ax.get_legend()
    print(legend)
    if legend:
        print("legend exists")
        legend.remove()
    
    ax2.get_legend()
    legend = ax2.get_legend()
    print(legend)
    if legend:
        print("legend2 exists")
        legend.remove()


def plot_future(df, category = "temperature_2m")->Response:
    plt.figure(figsize=(10,5))
    
    ax = df.plot(x='measuredAt', y='levelAtHour', marker='o', markersize=4, color='#3F76B8', label = 'Water Level')

    ax2 = df.plot(x='measuredAt', y = category, markersize=4, ax = ax, secondary_y = True, color = mapping(category,"colors"), marker='o', label = mapping(category, "legend"))
    
    customization(df, category, ax, ax2)
    #ax.spines['top'].set_visible(False)

    return getResponseImage(plt)

def plot_past(df, category = 'temperature_2m')->Response:
    plt.figure()
    ax = df.plot(
        kind = 'line', 
        x='measuredAt', 
        y='levelAtHour', 
        marker = 'o',
        markersize = .5,
        color='#3F76B8'
    )

    ax2 = df.plot(
        kind = 'line',
        x = 'measuredAt',
        y = category,
        color = mapping(category,"colors"),
        marker = 'o',
        markersize = 1,
        secondary_y = True,
        label = mapping(category, "legend"),
        ax = ax
    )

    customization(df, category, ax, ax2)

    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')

    return getResponseImage(plt)

def getResponseImage(plt):
    buffer = io.BytesIO()

    plt.savefig(buffer, bbox_inches="tight", format="png")
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
