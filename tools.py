from mcp.server.fastmcp import FastMCP
import requests
from datetime import datetime
import time
import os

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



if __name__ == "__main__":
    # Remove try/except - let FastMCP handle errors properly
    # Don't print anything to stdout - it breaks JSON-RPC
    mcp.run(transport="stdio")
