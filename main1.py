import time
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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
        time.sleep(40)
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


def get24hData():
    PL_datetime_now = (datetime.datetime.today())
    PL_datetime_24h_earlier = PL_datetime_now - datetime.timedelta(hours=24)

    PL_datetime_now_day = PL_datetime_now.date().day
    PL_datetime_now_month = PL_datetime_now.date().month
    PL_datetime_now_year = PL_datetime_now.date().year
    PL_datetime_24h_earlier_day = PL_datetime_24h_earlier.date().day
    PL_datetime_24h_earlier_month = PL_datetime_24h_earlier.date().month
    PL_datetime_24h_earlier_year = PL_datetime_24h_earlier.date().year
    PL_datetime_now_hour = PL_datetime_now.time().hour
    PL_datetime_24h_earlier_hour = PL_datetime_24h_earlier.time().hour

    adapted_PL_datetime_now = datetime.datetime(PL_datetime_now_year, PL_datetime_now_month, PL_datetime_now_day,PL_datetime_now_hour, 0 )
    adapted_PL_PL_datetime_24h_earlier = datetime.datetime(PL_datetime_24h_earlier_year,PL_datetime_24h_earlier_month,PL_datetime_24h_earlier_day,PL_datetime_24h_earlier_hour, 0 )

    return getHistoricDataOnly20Days(adapted_PL_PL_datetime_24h_earlier, adapted_PL_datetime_now)


#Stała kolejność kolumn, jaką chcesz uzyskać
column_order = [
    'Date',
    'PkRzeszPilsu-PM2.5-1g',
    'PkRzeszPilsu-PM10-1g',
    'PkRzeszPilsu-CO-1g',
    'PkRzeszPilsu-NO2-1g',
    'PkRzeszRejta-O3-1g',
    'PkRzeszRejta-SO2-1g']

def reorder_columns(df, column_order):
    # Upewnij się, że DataFrame ma te same kolumny, co column_order
    df = df[column_order]
    return df

#df_today = getTodaysData()
#df_today = reorder_columns(df_today, column_order)
#df_today.to_csv('TodaysData.csv', index=False)
#df_24h_earlier = get24hData()
#df_24h_earlier = reorder_columns(df_24h_earlier, column_order)
#df_24h_earlier.to_csv('24h.csv', index=False)

#df_historic=getHistoricDataOnly20Days("2025-02-01 00:00", "2025-02-19 00:00")
#df_historic = reorder_columns(df_historic, column_order)
#df_historic.to_csv('HistoricDataOnly20Days.csv', index=False)

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

def caqi_co_without_changing_units(value_mg):
    if pd.isna(value_mg): return None
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
    elif value <= 75: return "Średni"
    elif value <= 100: return "Zły"
    else: return "Bardzo zły"

def add_caqi_column(df):
    # Zrób kopię dataframe tylko do obliczeń
    df_temp = df.copy()

    # Stwórz nowe kolumny tymczasowe z oryginalnych danych
    for original_col, short_name in sensor_column_map.items():
        if original_col in df_temp.columns:
            df_temp[short_name] = pd.to_numeric(df_temp[original_col], errors='coerce')

    # Oblicz CAQI na podstawie tymczasowych kolumn
    df_temp['CAQI'] = df_temp.apply(compute_caqi_row, axis=1)
    df_temp['CAQI_Opis'] = df_temp['CAQI'].apply(caqi_description)

    # Teraz wróć do oryginalnego dataframe + tylko dwie nowe kolumny
    df['CAQI'] = df_temp['CAQI']
    df['CAQI_Opis'] = df_temp['CAQI_Opis']

    return df

#df_today_with_caqi = add_caqi_column(df_today)
#df_today_with_caqi.to_csv("TodaysData_with_CAQI.csv", index=False)
#df_24h_earlier_with_caqi = add_caqi_column(df_24h_earlier)
#df_24h_earlier_with_caqi.to_csv("24h_with_CAQI.csv", index=False)


#df_historic_with_caqi=add_caqi_column(df_historic)
#df_historic_with_caqi.to_csv("HistoricData_with_CAQI.csv", index=False)

pollutions = ["PM2.5", "PM10", "CO", "NO2", "O3", "SO2"]

def draw_caqi_background(ax):
    caqi_ranges = [
        (0, 25, 'green', 'Bardzo dobry'),
        (25, 50, 'yellow', 'Dobry'),
        (50, 75, 'orange', 'Średni'),
        (75, 100, 'red', 'Zły'),
        (100, 500, 'purple', 'Bardzo zły'),
    ]
    for lower, upper, color, label in caqi_ranges:
        ax.axhspan(lower, upper, facecolor=color, alpha=0.2)

# Funkcja pomocnicza - znajdowanie kolumny pasującej do danego typu zanieczyszczenia
def find_pollution_column(data, pollution):
    for col in data.columns:
        if pollution in col:
            return col
    return None

def TodaysCAQIPlot(data):
    data['Date'] = pd.to_datetime(data['Date'])

    for pollution in pollutions:
        col_name = find_pollution_column(data, pollution)
        if col_name is None:
            print(f"Nie znaleziono kolumny dla {pollution}.")
            continue

        data[col_name] = pd.to_numeric(data[col_name], errors='coerce')

        hours = []
        caqi_values = []
        for i, temp in enumerate(data['Date']):
            hours.append(temp.hour)

            value = data[col_name].iloc[i]
            if pd.isna(value):
                caqi_values.append(np.nan)
            else:
                if pollution == 'PM2.5':
                    caqi_values.append(caqi_pm25(value))
                elif pollution == 'PM10':
                    caqi_values.append(caqi_pm10(value))
                elif pollution == 'CO':
                    caqi_values.append(caqi_co(value))
                elif pollution == 'NO2':
                    caqi_values.append(caqi_no2(value))
                elif pollution == 'O3':
                    caqi_values.append(caqi_o3(value))
                elif pollution == 'SO2':
                    caqi_values.append(caqi_so2(value))

        plot_data = pd.DataFrame({'Hour': hours, 'CAQI': caqi_values})
        plot_data = plot_data.dropna(subset=['CAQI'])

        if plot_data.empty:
            print(f"Brak danych do wyświetlenia dla {pollution} po usunięciu NaN.")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(plot_data['Hour'], plot_data['CAQI'], marker='o', linestyle='-', markersize=5, color='blue')

        plt.xticks(rotation=45)
        plt.xticks(range(0, 24, 2))

        plt.title(f"Godzinowy CAQI ({pollution}) - {data['Date'].iloc[0].strftime('%Y-%m-%d')}")
        plt.xlabel("Godzina")
        plt.ylabel("CAQI")

        max_caqi = plot_data['CAQI'].max()
        if not np.isnan(max_caqi):
            plt.ylim(0, max_caqi + 5)
        else:
            plt.ylim(0, 100)

        draw_caqi_background(plt)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def CAQIPlot24h(data):
    data['Date'] = pd.to_datetime(data['Date'])

    start_time = data['Date'].min()
    end_time = start_time + pd.Timedelta(hours=24)

    # Filtrowanie od startu przez 24h
    data = data[(data['Date'] >= start_time) & (data['Date'] < end_time)]

    for pollution in pollutions:
        col_name = find_pollution_column(data, pollution)
        if col_name is None:
            print(f"Nie znaleziono kolumny dla {pollution}.")
            continue

        data[col_name] = pd.to_numeric(data[col_name], errors='coerce')

        times = []
        caqi_values = []
        for i, temp in enumerate(data['Date']):
            value = data[col_name].iloc[i]

            if pd.isna(value):
                continue

            if pollution == 'PM2.5':
                caqi = caqi_pm25(value)
            elif pollution == 'PM10':
                caqi = caqi_pm10(value)
            elif pollution == 'CO':
                caqi = caqi_co(value)
            elif pollution == 'NO2':
                caqi = caqi_no2(value)
            elif pollution == 'O3':
                caqi = caqi_o3(value)
            elif pollution == 'SO2':
                caqi = caqi_so2(value)
            else:
                continue

            times.append(temp)
            caqi_values.append(caqi)

        if not times:
            print(f"Brak danych do wyświetlenia dla {pollution}.")
            continue

        plot_data = pd.DataFrame({'Time': times, 'CAQI': caqi_values})
        plot_data = plot_data.sort_values('Time')

        # Wykres
        plt.figure(figsize=(12, 6))
        plt.plot(plot_data['Time'], plot_data['CAQI'], marker='o', linestyle='-', markersize=5, color='blue')

        plt.title(
            f"Godzinowy CAQI ({pollution}) - {start_time.strftime('%Y-%m-%d %H:%M')} do {end_time.strftime('%Y-%m-%d %H:%M')}")
        plt.xlabel("Godzina")
        plt.ylabel("CAQI")

        max_caqi = plot_data['CAQI'].max()
        plt.ylim(0, max(100, max_caqi + 5))

        draw_caqi_background(plt)
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



file_path1 = '24h_with_CAQI.csv'
#file_path2 = 'TodaysData_with_CAQI.csv'

# Wczytanie pliku CSV do DataFrame
h = pd.read_csv(file_path1)
#d = pd.read_csv(file_path2)



def HistoricCAQIPlot(data):
    data['Date'] = pd.to_datetime(data['Date'])

    for pollution in pollutions:
        col_name = find_pollution_column(data, pollution)
        if col_name is None:
            print(f"Nie znaleziono kolumny dla {pollution}.")
            continue

        data[col_name] = pd.to_numeric(data[col_name], errors='coerce')

        caqi_values = []
        for value in data[col_name]:
            if pd.isna(value):
                caqi_values.append(np.nan)
            else:
                if pollution == 'PM2.5':
                    caqi_values.append(caqi_pm25(value))
                elif pollution == 'PM10':
                    caqi_values.append(caqi_pm10(value))
                elif pollution == 'CO':
                    caqi_values.append(caqi_co_without_changing_units(value))
                elif pollution == 'NO2':
                    caqi_values.append(caqi_no2(value))
                elif pollution == 'O3':
                    caqi_values.append(caqi_o3(value))
                elif pollution == 'SO2':
                    caqi_values.append(caqi_so2(value))

        data['CAQI_' + pollution] = caqi_values

        daily_avg = data.groupby(data['Date'].dt.date)['CAQI_' + pollution].mean().reset_index()
        daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
        daily_avg = daily_avg.dropna(subset=['CAQI_' + pollution])

        if daily_avg.empty:
            print(f"Brak danych do wyświetlenia dla {pollution} po usunięciu NaN.")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(daily_avg['Date'], daily_avg['CAQI_' + pollution], marker='o', linestyle='-', markersize=5, color='blue')

        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.xticks(rotation=45)

        plt.title(f"Średni dzienny CAQI ({pollution})")
        plt.xlabel("Data")
        plt.ylabel("Średni dzienny CAQI")

        max_caqi = daily_avg['CAQI_' + pollution].max()
        if not np.isnan(max_caqi):
            plt.ylim(0, max_caqi + 5)
        else:
            plt.ylim(0, 100)

        draw_caqi_background(plt)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Wywołanie funkcji
#HistoricCAQIPlot(h)
CAQIPlot24h(h)