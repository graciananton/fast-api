import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timezone, timedelta

load_dotenv() 

STATIONS_URL = os.getenv("STATIONS_URL")

def get_stations():
    res = requests.get(STATIONS_URL)
    stations = res.json()
    return stations

def days_ago(days:int):
    time_now = datetime.now(timezone.utc)
    time_ago = time_now - timedelta(days = days)
    formatted_time_ago = time_ago.strftime("%Y-%m-%d %H:%M:%S.%f")
    return formatted_time_ago