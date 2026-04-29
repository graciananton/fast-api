from fastapi import FastAPI
from datetime import datetime, timezone, timedelta
from zlib import crc32
import matplotlib.pyplot as plt
import joblib
from scipy.stats import randint

from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import pandas as pd

import requests
import json

from fastapi import FastAPI
from api.routes import router

app = FastAPI()

app.include_router(router)