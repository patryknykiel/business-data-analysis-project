import time
import requests
import datetime
import pandas as pd
from datetime import timedelta


api_key = "382BF27FHEE7GQ5Q2HNQJ83E5"



def getTodaysVisualCrossingData(api_key: str, location: str = "Rzeszow,PL"):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/today"
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": api_key,
        "contentType": "json"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Najpierw próbujemy wziąć tylko 'obs'
    hours = [hour for hour in data["days"][0]["hours"] if hour.get("source") == "obs"]

    # Jeśli nie ma danych 'obs', bierzemy wszystkie godziny
    if not hours:
        print("No 'obs' data found. Using all hourly data instead.")
        hours = data["days"][0]["hours"]

    df = pd.json_normalize(hours)

    # Sprawdźmy jaka jest nazwa kolumny z czasem
    datetime_column = None
    for col in df.columns:
        if "datetime" in col and not col.startswith("datetimeEpoch"):
            datetime_column = col
            break

    if datetime_column:
        date = data["days"][0]["datetime"]
        df["Date"] = pd.to_datetime(date + " " + df[datetime_column])
        df.drop(columns=[datetime_column], inplace=True)
        df = df[["Date"] + [col for col in df.columns if col != "Date"]]

        df.to_csv("visualCrossing_today_obs.csv", index=False)
        print("Saved weather data to visualCrossing_today_obs.csv")
    else:
        print("No datetime column found in data.")
def getHistoricDataOnly20DaysVisualCrossing(api_key: str, location: str = "Rzeszow,PL"):
    end_date = datetime.datetime.now().date()
    start_date = end_date - timedelta(days=19)

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_date}/{end_date}"
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": api_key,
        "contentType": "json"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    all_obs_rows = []

    for day in data["days"]:
        date = day["datetime"]
        for hour in day.get("hours", []):
            if hour.get("source") == "obs":
                full_datetime = f"{date} {hour['datetime']}"
                hour_data = hour.copy()
                hour_data["Date"] = pd.to_datetime(full_datetime)
                del hour_data["datetime"]
                all_obs_rows.append(hour_data)

    df = pd.DataFrame(all_obs_rows)
    df = df[["Date"] + [col for col in df.columns if col != "Date"]]

    df.to_csv("visualCrossing_past20days_obs.csv", index=False)
    print("Saved past 20 days of observed data to visualCrossing_past20days_obs.csv")

getTodaysVisualCrossingData(api_key)
getHistoricDataOnly20DaysVisualCrossing(api_key)