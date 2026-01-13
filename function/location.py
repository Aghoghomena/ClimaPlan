import time
from datetime import datetime

#get the users location using ipinfo.io and return it as a dictionary processing the loc field to get latitude and longitude
def get_user_location():
    import requests
    

    response = requests.get("https://ipinfo.io/json")
    if response.status_code == 200:
        data = response.json()
        loc = data.get("loc")
        latitude, longitude = loc.split(",") if loc else (None, None)
        location_info = {
            "ip": data.get("ip"),
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "Local time:": datetime.now(),
            "Timezone": time.tzname,
            "latitude": float(latitude),
            "longitude": float(longitude)
        }
        return location_info
    else:
        return None