from mcp.server.fastmcp import FastMCP
import requests
from datetime import datetime
import time
import os
import pandas as pd

mcp = FastMCP("climate_report")

#get the users location using ipinfo.io and return it as a dictionary processing the loc field to get latitude and longitude
@mcp.tool()
def get_user_location() -> dict:
    """
    Fetch location of the user using ipinfo.io and return dict with lat/lon parsed from loc.
    Always returns a dict (never empty). On error, returns {"error": "...", "details": "..."}.
    """
    token = os.getenv("IPINFO_TOKEN")  # optional but recommended

    url = "https://ipinfo.io/json"
    params = {"token": token} if token else None

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        loc = data.get("loc")  # "lat,lon"
        lat = lon = None
        if loc and "," in loc:
            lat_str, lon_str = loc.split(",", 1)
            lat, lon = float(lat_str), float(lon_str)

        return {
            "ip": data.get("ip"),
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "local_time": datetime.now().isoformat(),
            "timezone": time.tzname[0] if getattr(time, "tzname", None) else "UTC",
            "latitude": lat,
            "longitude": lon,
            "raw": data,  # helpful for debugging; remove later if you want
        }

    except Exception as e:
        return {
            "error": "Unable to fetch location from ipinfo",
            "details": str(e),
        }


@mcp.tool()
def get_weather_for_today(latitude: float, longitude: float) -> dict:
    """
    Fetches weather data for the given latitude and longitude using Open-Meteo API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        dict: Weather data including temperature, wind speed, and weather code.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("current_weather", {})
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def filter_historical_data(historical_data: dict) -> dict:
    """
    Processes historical weather data to compute average temperature and total precipitation.

    Args:
        historical_data (dict): Raw historical weather data.
    Returns:
        dict: Processed data with the baseline weather data needed.
    """
    try:
        weather_list = []
        daily = historical_data["daily"]

        for date, tmax, tmin, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_sum"],
        ):
            weather_list.append({
                "date": date,
                "temp_max": tmax,
                "temp_min": tmin,
                "precipitation": rain,
            })
           # Create DataFrame and get day of year
        df = pd.DataFrame(weather_list)
        df["date"] = pd.to_datetime(df["date"])
        df["doy"] = df["date"].dt.dayofyear

        # Create baseline data window
        data_window = 5
        current_date = datetime.now().strftime('%Y-%m-%d')
        target_doy = pd.Timestamp(current_date).dayofyear
        lower = target_doy - data_window
        upper = target_doy + data_window

        if lower < 1:
            baseline = df[(df["doy"] >= 366 + lower) | (df["doy"] <= upper)]
        elif upper > 366:
            baseline = df[(df["doy"] >= lower) | (df["doy"] <= upper - 366)]
        else:
            baseline = df[(df["doy"] >= lower) & (df["doy"] <= upper)]
        return baseline.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
