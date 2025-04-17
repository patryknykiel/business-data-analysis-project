import time
import requests
import datetime
import pandas as pd

'''
PkRzeszPilsu-PM2.5-1 - 20277
PkRzeszPilsu-PM10-1g - 16344
PkRzeszPilsu-CO-1g - 16343
PkRzeszPilsu-NO2-1g - 16319
PkRzeszRejta-O3-1g -4385
PkRzeszRejta-SO2-1 - 4391
'''

apiLinks = ["https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/20277", "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16344",
            "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16343", "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16319",
            "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/4385", "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/4391"]


historicApiLinks = ["https://api.gios.gov.pl/pjp-api/v1/rest/archivalData/getDataBySensor/20277",
                    "https://api.gios.gov.pl/pjp-api/v1/rest/archivalData/getDataBySensor/16343",
                    "https://api.gios.gov.pl/pjp-api/v1/rest/archivalData/getDataBySensor/4391",
                    "https://api.gios.gov.pl/pjp-api/v1/rest/archivalData/getDataBySensor/16344",
                    "https://api.gios.gov.pl/pjp-api/v1/rest/archivalData/getDataBySensor/4385",
                    "https://api.gios.gov.pl/pjp-api/v1/rest/archivalData/getDataBySensor/16319"]


def getTodaysData():
    todayData = {
        "Date": [],
    }
    for sensor in apiLinks:
        dateList = []
        valueList = []
        response = requests.get(sensor + "?size=24").json()
        data = response['Lista danych pomiarowych']
        for i in range(len(data)):
            if pd.to_datetime(data[i]['Data']).day == datetime.datetime.now().day:
                dateList.append(data[i]['Data'])
                valueList.append(data[i]['Wartość'])

            if len(todayData["Date"]) == 0:
                todayData.update({"Date": dateList})
            todayData.update({data[i]['Kod stanowiska']: valueList})

    return pd.DataFrame(todayData).sort_values(by=['Date'])

def getHistoricDataOnly20Days(dateFrom, dateTo):

    dateF = pd.to_datetime(dateFrom)
    dateT = pd.to_datetime(dateTo)
    dateFromHour = dateF.time().hour
    dateFromMinute = dateF.time().minute
    dateToHour = dateT.time().hour
    dateToMinute = dateT.time().minute

    if (dateT - dateF).days > 20:
        return "Date range must be less or equal 20 days!! Try again."

    if dateF.time().hour < 10:
        dateFromHour = "0" + str(dateF.time().hour)

    if dateF.time().minute < 10:
        dateFromMinute = "0" + str(dateF.time().minute)

    if dateT.time().hour < 10:
        dateToHour = "0" + str(dateT.time().hour)

    if dateF.time().minute < 10:
        dateToMinute = "0" + str(dateT.time().minute)

    historicData = {}

    dateList = []
    dateList.append(str(dateF))

    k = 0

    while k < (dateT - dateF).total_seconds() / 3600:
        dateList.insert(k + 1, str(pd.to_datetime(dateList[k]) + datetime.timedelta(hours=1)))
        k += 1

    historicData.update({"Date": dateList})

    for sensor in historicApiLinks:
        valueList = []
        time.sleep(20)
        response = requests.get(f'{sensor}?size=500&dateFrom={dateF.date()}%20{dateFromHour}%3A{dateFromMinute}'
                                f'&dateTo={dateT.date()}%20{dateToHour}%3A{dateToMinute}').json()

        data = response['Lista archiwalnych wyników pomiarów']

        for date in dateList:
            for i in range(len(data)):
                if date == (data[i]['Data']):
                    valueList.append(data[i]['Wartość'])
                    break
                elif i == len(data)-1:
                    valueList.append("")

        historicData.update({data[i]['Kod stanowiska']: valueList})

    return pd.DataFrame(historicData)


df_today = getTodaysData()
df_historic=getHistoricDataOnly20Days("2023-01-01 00:00", "2023-01-19 00:00")

df_historic.to_csv('HistoricDataOnly20Days.csv', index=False)
df_today.to_csv('TodaysData.csv', index=False)


def caqi_pm10(value):
    if pd.isna(value): return None
    if value <= 25: return value
    elif value <= 50: return 25 + (value - 25)
    elif value <= 90: return 50 + (value - 50) * 0.625
    elif value <= 180: return 75 + (value - 90) * 0.278
    else: return 100

def caqi_pm25(value):
    if pd.isna(value): return None
    if value <= 15: return value * (25 / 15)
    elif value <= 30: return 25 + (value - 15) * (25 / 15)
    elif value <= 55: return 50 + (value - 30) * (25 / 25)
    elif value <= 110: return 75 + (value - 55) * (25 / 55)
    else: return 100

def caqi_no2(value):
    if pd.isna(value): return None
    if value <= 50: return value * 0.5
    elif value <= 100: return 25 + (value - 50) * 0.5
    elif value <= 200: return 50 + (value - 100) * 0.25
    elif value <= 400: return 75 + (value - 200) * 0.125
    else: return 100

def caqi_o3(value):
    if pd.isna(value): return None
    if value <= 60: return value * 25 / 60
    elif value <= 120: return 25 + (value - 60) * 25 / 60
    elif value <= 180: return 50 + (value - 120) * 25 / 60
    elif value <= 240: return 75 + (value - 180) * 25 / 60
    else: return 100

def caqi_co(value):
    if pd.isna(value): return None
    value_mg = value / 1000  # zamiana µg/m³ -> mg/m³
    if value_mg <= 5: return value_mg * 5
    elif value_mg <= 10: return 25 + (value_mg - 5) * 5
    elif value_mg <= 35: return 50 + (value_mg - 10) * (25 / 25)
    elif value_mg <= 60: return 75 + (value_mg - 35) * (25 / 25)
    else: return 100

def caqi_so2(value):
    if pd.isna(value): return None
    if value <= 50: return value * 0.5
    elif value <= 100: return 25 + (value - 50) * 0.5
    elif value <= 350: return 50 + (value - 100) * 0.1
    elif value <= 500: return 75 + (value - 350) * 0.167
    else: return 100


sensor_column_map = {
    'PkRzeszPilsu-PM2.5-1g': 'PM2.5',
    'PkRzeszPilsu-PM10-1g': 'PM10',
    'PkRzeszPilsu-CO-1g': 'CO',
    'PkRzeszPilsu-NO2-1g': 'NO2',
    'PkRzeszRejta-O3-1g': 'O3',
    'PkRzeszRejta-SO2-1g': 'SO2'
}

def compute_caqi_row(row):
    values = []
    if 'PM2.5' in row: values.append(caqi_pm25(row['PM2.5']))
    if 'PM10' in row: values.append(caqi_pm10(row['PM10']))
    if 'CO' in row: values.append(caqi_co(row['CO']))
    if 'NO2' in row: values.append(caqi_no2(row['NO2']))
    if 'O3' in row: values.append(caqi_o3(row['O3']))
    if 'SO2' in row: values.append(caqi_so2(row['SO2']))
    values = [v for v in values if v is not None]
    return max(values) if values else None

def caqi_description(value):
    if pd.isna(value): return None
    if value <= 25: return "Bardzo dobry"
    elif value <= 50: return "Dobry"
    elif value <= 75: return "Umiarkowany"
    elif value <= 100: return "Dostateczny"
    elif value <= 125: return "Zły"
    else: return "Bardzo zły"

def add_caqi_column(df):
    df = df.rename(columns=sensor_column_map)

    for col in sensor_column_map.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['CAQI'] = df.apply(compute_caqi_row, axis=1)
    df['CAQI_Opis'] = df['CAQI'].apply(caqi_description)
    return df

df_today_with_caqi = add_caqi_column(df_today)
df_today_with_caqi.to_csv("TodaysData_with_CAQI.csv", index=False)

df_historic_with_caqi=add_caqi_column(df_historic)
df_historic_with_caqi.to_csv("HistoricData_with_CAQI.csv", index=False)