import time
from datetime import datetime

def get_user_location():
    import requests
    

    response = requests.get("https://ipinfo.io/json")
    if response.status_code == 200:
        data = response.json()
        location_info = {
            "ip": data.get("ip"),
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country"),
            "Local time:": datetime.now(),
            "Timezone": time.tzname
        }
        return location_info
    else:
        return None