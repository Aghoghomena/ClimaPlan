# from ast import main
# from xmlrpc import client
# import sys as system
# from langchain_mcp_adapters.client import MultiServerMCPClient
# import asyncio

# mcp_client  = MultiServerMCPClient(
#     {
#         "climate_report_srv": {
#             "transport": "stdio",
#             "command": "uv",
#             "args": ["run", "python", "tools.py"],
#         },
#          "open_meteo": {
#             "transport": "stdio",
#             "command": "npx",
#             "args": ["open-meteo-mcp-server"],
#             "env": {
#                 "OPEN_METEO_ARCHIVE_API_URL": "https://archive-api.open-meteo.com/v1/archive",
#             }
#         }

#     }
# )
# async def get_tools():
#     alltools = await mcp_client.get_tools()
#     print(f"Found {len(alltools)} total tools:")
#     ALLOW = {"weather_archive", "geocoding", "get_user_location", "get_weather_for_today"}  # only what you need
#     tools= [t for t in alltools if t.name in ALLOW]
#     print(f"filtered {len(tools)} total tools:")

#     print(tools)

# if __name__ == "__main__":
#     asyncio.run(get_tools())



import requests
import pandas as pd
from datetime import datetime


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 53.3331,
    "longitude": -6.2489,
    "start_date": "2016-01-12",
    "end_date": "2026-01-12",
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,shortwave_radiation_sum",
    "timezone": "Europe/Dublin",
}

r = requests.get(url, params=params, timeout=30)
r.raise_for_status()
data = r.json()

#print(data.get("daily", {}))
# Extract relevant daily weather data
weather_list = []
daily = data["daily"]

for date, tmax, tmin, rain, wind, shortwave in zip(
    daily["time"],
    daily["temperature_2m_max"],
    daily["temperature_2m_min"],
    daily["precipitation_sum"],
    daily["windspeed_10m_max"],
    daily["shortwave_radiation_sum"],
):
    weather_list.append({
        "date": date,
        "temp_max": tmax,
        "temp_min": tmin,
        "precipitation": rain,
        "windspeed": wind,
        "shortwave": shortwave
    })



df = pd.DataFrame(weather_list)

df["date"] = pd.to_datetime(df["date"])
df["doy"] = df["date"].dt.dayofyear

data_window = 5
target_doy = pd.Timestamp("2026-01-12").dayofyear
lower = target_doy - data_window
upper = target_doy + data_window

if lower < 1:
    baseline = df[(df["doy"] >= 366 + lower) | (df["doy"] <= upper)]
elif upper > 366:
    baseline = df[(df["doy"] >= lower) | (df["doy"] <= upper - 366)]
else:
    baseline = df[(df["doy"] >= lower) & (df["doy"] <= upper)]

print(baseline)


