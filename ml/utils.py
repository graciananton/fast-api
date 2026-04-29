import requests
from dotenv import load_dotenv
import os

load_dotenv() 

STATIONS_URL = os.getenv("STATIONS_URL")

def get_stations():
    res = requests.get(STATIONS_URL)
    stations = res.json()
    return stations

