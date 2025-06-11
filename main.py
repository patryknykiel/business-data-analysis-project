import time
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.tree import DecisionTreeRegressor
import os
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
        time.sleep(30)
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
    todayData = {
        "Date": [],
    }
    for sensor in apiLinks:
        dateList = []
        valueList = []
        response = requests.get(sensor + "?size=25").json()
        data = response['Lista danych pomiarowych']
        for i in range(len(data)):
            dateList.append(data[i]['Data'])
            valueList.append(data[i]['Wartość'])

            if len(todayData["Date"]) == 0:
                todayData.update({"Date": dateList})
            todayData.update({data[i]['Kod stanowiska']: valueList})

    return pd.DataFrame(todayData).sort_values(by=['Date'])


# Stała kolejność kolumn, jaką chcesz uzyskać
column_order = [
    'Date',
    'PkRzeszPilsu-PM2.5-1g',
    'PkRzeszPilsu-PM10-1g',
    'PkRzeszPilsu-CO-1g',
    'PkRzeszPilsu-NO2-1g',
    'PkRzeszRejta-O3-1g',
    'PkRzeszRejta-SO2-1g'
]

def reorder_columns(df, column_order):
    # Upewnij się, że DataFrame ma te same kolumny, co column_order
    df = df[column_order]
    return df



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
def add_caqi_legend_custom(ax):
    import matplotlib.patches as mpatches

    labels_colors = [
        ('Bardzo dobry (0–25)', (0, 1, 0, 0.2)),     # zielony
        ('Dobry (25–50)', (1, 1, 0, 0.2)),           # żółty
        ('Średni (50–75)', (1, 0.65, 0, 0.2)),       # pomarańczowy
        ('Zły (75–100)', (1, 0, 0, 0.2)),            # czerwony
        ('Bardzo zły (100+)', (0.5, 0, 0.5, 0.2)),   # fioletowy
    ]

    patches = [mpatches.Patch(color=color, label=label) for label, color in labels_colors]
    ax.legend(handles=patches, loc='upper right', title="Skala CAQI")


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

        draw_caqi_background(plt.gca())
        add_caqi_legend_custom(plt.gca())
        plt.grid(True)
        plt.tight_layout()
        plt.show()



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

        draw_caqi_background(plt.gca())
        add_caqi_legend_custom(plt.gca())
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def CAQIPlot24h(data):
    data['Date'] = pd.to_datetime(data['Date'])

    start_time = data['Date'].min()
    end_time = start_time + pd.Timedelta(hours=24)

    # Filtrowanie od startu przez 24h
    data = data[(data['Date'] >= start_time) & (data['Date'] < end_time)].copy()

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

        draw_caqi_background(plt.gca())
        add_caqi_legend_custom(plt.gca())
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



df_today = getTodaysData()
df_today = reorder_columns(df_today, column_order)
df_today.to_csv('TodaysData.csv', index=False)


#df_historic=getHistoricDataOnly20Days("2025-01-01 00:00", "2025-01-15 00:00")
#df_historic = reorder_columns(df_historic, column_order)
#df_historic.to_csv('HistoricDataOnly20Days.csv', index=False)

df_24h=get24hData()
df_24h=reorder_columns(df_24h, column_order)
df_24h.to_csv('24HData.csv', index=False)


df_today_with_caqi = add_caqi_column(df_today)
df_today_with_caqi.to_csv("TodaysData_with_CAQI.csv", index=False)


#df_historic_with_caqi=add_caqi_column(df_historic)
#df_historic_with_caqi.to_csv("HistoricData_with_CAQI.csv", index=False)

df_24h_with_caqi=add_caqi_column(df_24h)
df_24h_with_caqi.to_csv("24HData_with_CAQI.csv",index=False)


# Wczytanie pliku CSV do DataFrame
historic = pd.read_csv("HistoricData_with_CAQI.csv")
today = pd.read_csv("TodaysData_with_CAQI.csv")
last24h = pd.read_csv("24HData_with_CAQI.csv")
year2023=pd.read_csv("2023_data_with_CAQI.csv")
weather=pd.read_csv("merged_energy_pollution_weather_arcus_6m.csv")



# Wywołanie funkcji
HistoricCAQIPlot(historic)
TodaysCAQIPlot(today)
CAQIPlot24h(last24h)

#NOWA CZĘŚĆ



historic_corr= pd.read_csv("HistoricData_with_CAQI.csv")

df_corr = historic_corr.drop(columns=['Date', 'CAQI_Opis'])
# Oblicz korelację
corr = df_corr.corr(method='pearson')  # Można też 'spearman'
# Heatmapa
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmapa korelacji między cechami (dane historyczne)")
plt.show()



df_corr = last24h.drop(columns=['Date', 'CAQI_Opis'])
#Oblicz korelację
corr = df_corr.corr(method='pearson')  # Można też 'spearman'
# Heatmapa
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmapa korelacji między cechami (dane 24h)")
plt.show()

df_corr = year2023.drop(columns=['Date'])
#Oblicz korelację
corr = df_corr.corr(method='pearson')  # Można też 'spearman'
# Heatmapa
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmapa korelacji między cechami (dane z 2023)")
plt.show()

df_corr = weather.drop(columns=['DayOfYear','DayOfWeek','Time','NOx','NO','Benzen'])
df_corr.to_csv("merged_energy_pollution_weather.csv",index=False)

#Oblicz korelację
corr = df_corr.corr(method='pearson')  # Można też 'spearman'
# Heatmapa
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmapa korelacji między cechami (dane o jakości powietrza + pogoda)")
plt.show()

#PREDICTION
# Wczytanie danych z pliku Excel
data = pd.read_excel("2023 dane.xlsx")

# Zamiana kolumny 'Date' na datę
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y %H:%M')

# Zastąpienie brakujących wartości metodą interpolacji
data.fillna(data.mean(), inplace=True)

def compute_caqi_row_train(row):
    values = []
    if 'PkRzeszPilsu-PM2.5-1g' in row: values.append(caqi_pm25(row['PkRzeszPilsu-PM2.5-1g']))
    if 'PkRzeszPilsu-PM10-1g' in row: values.append(caqi_pm10(row['PkRzeszPilsu-PM10-1g']))
    if 'PkRzeszPilsu-CO-1g' in row: values.append(caqi_co(row['PkRzeszPilsu-CO-1g']))  # CO w mikrogramach, nie trzeba przeliczać
    if 'PkRzeszPilsu-NO2-1g' in row: values.append(caqi_no2(row['PkRzeszPilsu-NO2-1g']))
    if 'PkRzeszRejta-O3-1g' in row: values.append(caqi_o3(row['PkRzeszRejta-O3-1g']))
    if 'PkRzeszRejta-SO2-1g' in row: values.append(caqi_so2(row['PkRzeszRejta-SO2-1g']))
    values = [v for v in values if v is not None]
    return max(values) if values else None

# Obliczanie CAQI dla każdej próbki
data['CAQI'] = data.apply(compute_caqi_row_train, axis=1)
data.to_csv('2023_data_with_CAQI.csv', index=False)

# Usunięcie wierszy z brakiem wartości CAQI (jeśli jakieś istnieją)
data = data.dropna(subset=['CAQI'])

# Zakładając, że cechy to zmienne zanieczyszczeń
features = ['PkRzeszPilsu-PM2.5-1g', 'PkRzeszPilsu-PM10-1g', 'PkRzeszPilsu-CO-1g',
            'PkRzeszPilsu-NO2-1g', 'PkRzeszRejta-O3-1g', 'PkRzeszRejta-SO2-1g']

#XGBoost
# Zmienna zależna to CAQI
X = data[features]
y = data['CAQI']

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stworzenie modelu regresyjnego XGBoost
model = xgb.XGBRegressor(eval_metric='rmse')

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# Ocena modelu - RMSE (manualnie obliczone)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}')

# Zapisanie modelu do pliku
model.save_model('xgboost_regressor.model')



#Ważność cech – pobranie z modelu
importance = model.feature_importances_
feature_names = features

# Stworzenie DataFrame z wynikami
importance_df = pd.DataFrame({
    'Cecha': feature_names,
    'Waznosc': importance
}).sort_values(by='Waznosc', ascending=False)

# Wyświetlenie tabeli w terminalu
print("\nWażność cech według XGBoost:")
print(importance_df)

# Wykres ważności cech
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Cecha'], importance_df['Waznosc'], color='skyblue')
plt.xlabel('Ważność')
plt.title('Ważność cech w modelu XGBoost')
plt.gca().invert_yaxis()  # Najważniejsze na górze
plt.tight_layout()
plt.savefig('XGBoost_Feature_Importance.png')  # Zapisz wykres do pliku
plt.show()


# Przewidywanie dla nowych danych
todays_data = pd.read_csv('TodaysData_with_CAQI.csv')

features = ['PkRzeszPilsu-PM2.5-1g', 'PkRzeszPilsu-PM10-1g', 'PkRzeszPilsu-CO-1g',
            'PkRzeszPilsu-NO2-1g', 'PkRzeszRejta-O3-1g', 'PkRzeszRejta-SO2-1g']


# Funkcja, która oblicza CAQI tylko na podstawie dostępnych danych
def predict_caqi(row):
    # Jeśli wszystkie wartości w wierszu są puste, zwróć NaN
    if row.isna().sum() == len(row):
        return np.nan

    # Sprawdzamy, czy są brakujące wartości w row
    missing_features = row.isna().sum()

    # Jeśli brakujące dane są, to można wypełnić brakujące wartości na przykład zerem
    if missing_features > 0:
        row = row.fillna(0)

    # Upewniamy się, że wiersz ma dokładnie 6 cech przed przewidywaniem
    if len(row) == 6:
        return model.predict([row])[0]
    else:
        return np.nan


# Przewidywanie wartości CAQI z uwzględnieniem braku zanieczyszczeń
todays_data['Predicted_CAQI'] = todays_data[features].apply(predict_caqi, axis=1)

# Dodanie kolumny Predicted_CAQI_Desc z opisami
todays_data['Predicted_CAQI_Desc'] = todays_data['Predicted_CAQI'].apply(caqi_description)

# Zapisanie wyników do pliku CSV
todays_data.to_csv('TodaysData_XGBoost_CAQI.csv', index=False)

print("Przewidywania zostały zapisane do pliku TodaysData_XGBoost_CAQI.csv")





# Przewidywanie dla danych historycznych
historic_data = pd.read_csv('HistoricData_with_CAQI.csv')

# Przewidywanie wartości CAQI z uwzględnieniem braku zanieczyszczeń
historic_data['Predicted_CAQI'] = historic_data[features].apply(predict_caqi, axis=1)

# Dodanie kolumny Predicted_CAQI_Desc z opisami
historic_data['Predicted_CAQI_Desc'] = historic_data['Predicted_CAQI'].apply(caqi_description)

# Zapisanie wyników do pliku CSV
historic_data.to_csv('HistoricData_XGBoost_CAQI.csv', index=False)

print("Przewidywania zostały zapisane do pliku HistoricData_XGBoost_CAQI.csv")


# Przewidywanie dla danych historycznych
last24h_data = pd.read_csv('24HData_with_CAQI.csv')

# Przewidywanie wartości CAQI z uwzględnieniem braku zanieczyszczeń
last24h_data['Predicted_CAQI'] = last24h_data[features].apply(predict_caqi, axis=1)

# Dodanie kolumny Predicted_CAQI_Desc z opisami
last24h_data['Predicted_CAQI_Desc'] = last24h_data['Predicted_CAQI'].apply(caqi_description)

# Zapisanie wyników do pliku CSV
last24h_data.to_csv('24HData_XGBoost_CAQI.csv', index=False)

print("Przewidywania zostały zapisane do pliku 24HData_XGBoost_CAQI.csv")

#RANDOM FOREST
X = data[features]
y = data['CAQI']

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie modelu Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

#Ważność cech – pobranie z modelu
importance = model.feature_importances_
feature_names = features

# Stworzenie DataFrame z wynikami
importance_df = pd.DataFrame({
    'Cecha': feature_names,
    'Waznosc': importance
}).sort_values(by='Waznosc', ascending=False)

# Wyświetlenie tabeli w terminalu / notebooku
print("\nWażność cech według Random Forest:")
print(importance_df)

# Wykres ważności cech
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Cecha'], importance_df['Waznosc'], color='lightgreen')
plt.xlabel('Ważność')
plt.title('Ważność cech w modelu Random Forest')
plt.gca().invert_yaxis()  # Najważniejsze na górze
plt.tight_layout()
plt.show()

# Ocena modelu - RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse:.2f}')

# Zapisanie modelu
joblib.dump(model, 'random_forest_regressor.model')

# Przewidywanie dla nowych danych
todays_data = pd.read_csv('TodaysData_with_CAQI.csv')
todays_data['Predicted_CAQI'] = model.predict(todays_data[features])

# Dodanie kolumny Predicted_CAQI_Desc z opisami
todays_data['Predicted_CAQI_Desc'] = todays_data['Predicted_CAQI'].apply(caqi_description)

# Zapisanie wyników do pliku CSV
todays_data.to_csv('TodaysData_RandomForest_CAQI.csv', index=False)
print("Przewidywania zostały zapisane do pliku TodaysData_RandomForest_CAQI.csv")

# Przewidywanie dla nowych danych
historic_data = pd.read_csv('HistoricData_with_CAQI.csv')
historic_data['Predicted_CAQI'] = model.predict(historic_data[features])

# Dodanie kolumny Predicted_CAQI_Desc z opisami
historic_data['Predicted_CAQI_Desc'] = historic_data['Predicted_CAQI'].apply(caqi_description)

# Zapisanie wyników do pliku CSV
historic_data.to_csv('HistoricData_RandomForest_CAQI.csv', index=False)
print("Przewidywania zostały zapisane do pliku HistoricData_RandomForest_CAQI.csv")


# Przewidywanie dla nowych danych
last24h_data = pd.read_csv('24HData_with_CAQI.csv')
last24h_data['Predicted_CAQI'] = model.predict(last24h_data[features])

# Dodanie kolumny Predicted_CAQI_Desc z opisami
last24h_data['Predicted_CAQI_Desc'] = last24h_data['Predicted_CAQI'].apply(caqi_description)

# Zapisanie wyników do pliku CSV
last24h_data.to_csv('24HData_RandomForest_CAQI.csv', index=False)
print("Przewidywania zostały zapisane do pliku 24HData_RandomForest_CAQI.csv")

# Usunięcie wierszy, gdzie brak wszystkich pomiarów
data = data.dropna(subset=features, how='all')

# Obliczenie CAQI
data['CAQI_PM2.5'] = data['PkRzeszPilsu-PM2.5-1g'].apply(caqi_pm25)
data['CAQI_PM10'] = data['PkRzeszPilsu-PM10-1g'].apply(caqi_pm10)
data['CAQI_NO2'] = data['PkRzeszPilsu-NO2-1g'].apply(caqi_no2)
data['CAQI_O3'] = data['PkRzeszRejta-O3-1g'].apply(caqi_o3)
data['CAQI_CO'] = data['PkRzeszPilsu-CO-1g'].apply(caqi_co)

# Użycie najwyższej wartości
data['CAQI'] = data[['CAQI_PM2.5', 'CAQI_PM10', 'CAQI_NO2', 'CAQI_O3', 'CAQI_CO']].max(axis=1)

# Teraz już masz CAQI, więc możesz trenować:
X = data[features].fillna(0)
y = data['CAQI']

#tree

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stworzenie i trenowanie modelu
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Predykcja na danych testowych
y_pred = tree_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'RMSE (Decision Tree): {rmse:.2f}')

# Zapis modelu
joblib.dump(tree_model, 'decision_tree_regressor.pkl')
print("Model drzewa decyzyjnego został zapisany.")

#Ważność cech – z drzewa decyzyjnego
importance = tree_model.feature_importances_
feature_names = features

# Tworzenie tabeli DataFrame
importance_df = pd.DataFrame({
    'Cecha': feature_names,
    'Waznosc': importance
}).sort_values(by='Waznosc', ascending=False)

# Wyświetlenie tabeli
print("\nWażność cech według Decision Tree:")
print(importance_df)

#Wykres ważności
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Cecha'], importance_df['Waznosc'], color='skyblue')
plt.xlabel('Ważność')
plt.title('Ważność cech w modelu Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Funkcja przewidująca
def predict_caqi(row):
    if row.isna().sum() == len(row):
        return np.nan
    row_filled = row.fillna(0)
    row_df = pd.DataFrame([row_filled], columns=features)  # <<< UWAGA: przekazanie DataFrame z nazwami cech
    return tree_model.predict(row_df)[0]

# Przewidywanie dla dzisiejszych danych
todays_data = pd.read_csv('TodaysData_with_CAQI.csv')

todays_data['Predicted_CAQI'] = todays_data[features].apply(predict_caqi, axis=1)
todays_data['Predicted_CAQI_Desc'] = todays_data['Predicted_CAQI'].apply(caqi_description)

todays_data.to_csv('TodaysData_DecisionTree_CAQI.csv', index=False)
print("Przewidywania dla Today's Data zapisane.")

# Przewidywanie dla danych historycznych
historic_data = pd.read_csv('HistoricData_with_CAQI.csv')

historic_data['Predicted_CAQI'] = historic_data[features].apply(predict_caqi, axis=1)
historic_data['Predicted_CAQI_Desc'] = historic_data['Predicted_CAQI'].apply(caqi_description)

historic_data.to_csv('HistoricData_DecisionTree_CAQI.csv', index=False)
print("Przewidywania dla danych historycznych zapisane.")



# Przewidywanie dla danych historycznych
last24h_data = pd.read_csv('24HData_with_CAQI.csv')

last24h_data['Predicted_CAQI'] = last24h_data[features].apply(predict_caqi, axis=1)
last24h_data['Predicted_CAQI_Desc'] = last24h_data['Predicted_CAQI'].apply(caqi_description)

last24h_data.to_csv('24HData_DecisionTree_CAQI.csv', index=False)
print("Przewidywania dla danych 24-godzinnych zapisane.")


def get_highest_caqi_component(data, output_file="caqi_analysis.txt", csv_file="caqi_analysis.csv"):
    real_data_result = None
    model_results = []

    last_valid_row = None
    for i in range(len(data) - 1, -1, -1):
        row = data.iloc[i]
        if any(not pd.isna(row[col]) for col in sensor_column_map.keys()):
            last_valid_row = row
            break

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Analiza CAQI z {timestamp} ===\n")

        if last_valid_row is None:
            f.write("Brak dostępnych danych do analizy (rzeczywiste pomiary).\n")
        else:
            caqi_values = {}
            concentrations = {}

            for full_col_name, short_name in sensor_column_map.items():
                value = pd.to_numeric(last_valid_row[full_col_name], errors='coerce')
                if pd.isna(value):
                    continue
                if short_name == 'PM2.5':
                    caqi = caqi_pm25(value)
                elif short_name == 'PM10':
                    caqi = caqi_pm10(value)
                elif short_name == 'CO':
                    caqi = caqi_co(value)
                elif short_name == 'NO2':
                    caqi = caqi_no2(value)
                elif short_name == 'O3':
                    caqi = caqi_o3(value)
                elif short_name == 'SO2':
                    caqi = caqi_so2(value)
                else:
                    caqi = None

                if caqi is not None:
                    caqi_values[short_name] = caqi
                    concentrations[short_name] = value

            if caqi_values:
                max_component = max(caqi_values, key=caqi_values.get)
                max_caqi = caqi_values[max_component]
                max_value = concentrations[max_component]
                caqi_desc = caqi_description(max_caqi)

                f.write("\n>>> Dane rzeczywiste (czujniki):\n")
                f.write(f"CAQI wynosi: {max_caqi:.2f} ({caqi_desc}).\n")
                f.write(f"\nNajwiększy wpływ ma składnik: {max_component}, stężenie: {max_value:.2f} µg/m³.\n")

                print("\n>>> Dane rzeczywiste (czujniki):")
                print(f"CAQI wynosi: {max_caqi:.2f} ({caqi_desc}).")
                print(f"\nNajwiększy wpływ ma składnik: {max_component}, stężenie: {max_value:.2f} µg/m³.")

                real_data_result = {
                    'Timestamp': timestamp,
                    'Source': 'Rzeczywiste',
                    'CAQI': max_caqi,
                    'Description': caqi_desc,
                    'Main_Pollutant': max_component,
                    'Concentration': max_value
                }
            else:
                f.write("Brak danych do obliczenia CAQI (rzeczywiste pomiary).\n")

        model_files = {
            "Drzewo decyzyjne": "TodaysData_DecisionTree_CAQI.csv",
            "XGBoost": "TodaysData_XGBoost_CAQI.csv",
            "Random Forest": "TodaysData_RandomForest_CAQI.csv"
        }

        for model_name, filename in model_files.items():
            try:
                model_data = pd.read_csv(filename)
                last_valid_row = None
                for i in range(len(model_data) - 1, -1, -1):
                    row = model_data.iloc[i]
                    if not pd.isna(row.get("Predicted_CAQI", None)):
                        last_valid_row = row
                        break

                f.write(f"\n>>> Model: {model_name}\n")
                print(f"\n>>> Model: {model_name}")

                if last_valid_row is not None:
                    caqi = last_valid_row['Predicted_CAQI']
                    desc = last_valid_row.get('Predicted_CAQI_Desc', 'brak opisu')
                    f.write(f"Przewidywane CAQI: {caqi:.2f} ({desc})\n")
                    print(f"Przewidywane CAQI: {caqi:.2f} ({desc})")
                else:
                    f.write("Brak danych do analizy (brak Predicted_CAQI).\n")
                    print("Brak danych do analizy (brak Predicted_CAQI).")

            except FileNotFoundError:
                f.write(f"\n>>> Model: {model_name}\n")
                f.write(f"Plik {filename} nie został znaleziony.\n")
                print(f"\n>>> Model: {model_name}")
                print(f"Plik {filename} nie został znaleziony.")
            except Exception as e:
                f.write(f"\n>>> Model: {model_name}\n")
                f.write(f"Wystąpił błąd podczas przetwarzania pliku {filename}: {e}\n")
                print(f"\n>>> Model: {model_name}")
                print(f"Wystąpił błąd podczas przetwarzania pliku {filename}: {e}")

    # Zapis do CSV tylko dla danych rzeczywistych
    if real_data_result:
        results_df = pd.DataFrame([real_data_result])
        results_df.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file), encoding='utf-8')

    print(f"\nWyniki zapisano do plików: {output_file} i {csv_file}")

get_highest_caqi_component(today)


# Definicja potencjalnych przyczyn dla zanieczyszczeń
pollution_causes = {
    'PM2.5': [
        "Spalanie paliw stałych w gospodarstwach domowych (np. węgiel, drewno)",
        "Emisje z pojazdów z silnikiem Diesla",
        "Procesy przemysłowe, takie jak spalanie w fabrykach",
        "Pył z placów budowy lub dróg"
    ],
    'PM10': [
        "Pył drogowy wzbijany przez ruch pojazdów",
        "Emisje z budownictwa i prac ziemnych",
        "Spalanie w piecach węglowych",
        "Naturalne źródła, takie jak pył z gleby lub piasku"
    ],
    'CO': [
        "Niekompletne spalanie paliw w pojazdach",
        "Ogrzewanie domowe (piece, kominki)",
        "Emisje z procesów przemysłowych",
        "Pożary lub spalanie odpadów"
    ],
    'NO2': [
        "Spaliny z pojazdów, szczególnie z silnikiem Diesla",
        "Emisje z elektrowni i zakładów przemysłowych",
        "Spalanie paliw kopalnych w kotłowniach",
        "Intensywny ruch drogowy w miastach"
    ],
    'O3': [
        "Reakcje fotochemiczne w obecności promieniowania słonecznego",
        "Wysokie temperatury i niska wilgotność",
        "Emisje prekursorów (NOx i lotne związki organiczne) z pojazdów i przemysłu",
        "Długotrwałe warunki bezwietrzne sprzyjające akumulacji ozonu"
    ],
    'SO2': [
        "Spalanie węgla w elektrowniach i zakładach przemysłowych",
        "Procesy przemysłowe, np. produkcja chemikaliów",
        "Emisje z rafinerii ropy naftowej",
        "Spalanie paliw siarkowych w kotłowniach"
    ]
}
def analyze_caqi_components(row, model, features, sensor_column_map):
    if row.isna().sum() == len(row):
        return None, None, None, None

    # Wypełnienie brakujących wartości zerami
    row_filled = row.fillna(0).infer_objects(copy=False)
    row_df = pd.DataFrame([row_filled], columns=features)

    # Przewidywanie CAQI
    predicted_caqi = model.predict(row_df)[0]
    caqi_desc = caqi_description(predicted_caqi)

    # Obliczenie CAQI dla każdego zanieczyszczenia
    caqi_values = {}
    concentrations = {}
    for full_col_name, short_name in sensor_column_map.items():
        value = pd.to_numeric(row[full_col_name], errors='coerce')
        if pd.isna(value):
            continue
        if short_name == 'PM2.5':
            caqi = caqi_pm25(value)
        elif short_name == 'PM10':
            caqi = caqi_pm10(value)
        elif short_name == 'CO':
            caqi = caqi_co(value)
        elif short_name == 'NO2':
            caqi = caqi_no2(value)
        elif short_name == 'O3':
            caqi = caqi_o3(value)
        elif short_name == 'SO2':
            caqi = caqi_so2(value)
        else:
            caqi = None

        if caqi is not None:
            caqi_values[short_name] = caqi
            concentrations[short_name] = value

    if not caqi_values:
        return None, None, None, None

    # Znalezienie zanieczyszczenia z największym CAQI
    max_component = max(caqi_values, key=caqi_values.get)
    max_caqi = caqi_values[max_component]
    max_value = concentrations[max_component]
    causes = pollution_causes.get(max_component, ["Brak informacji o przyczynach"])

    return max_component, max_caqi, max_value, causes

# Analiza ostatniego wiersza z danymi
features = ['PkRzeszPilsu-PM2.5-1g', 'PkRzeszPilsu-PM10-1g', 'PkRzeszPilsu-CO-1g',
            'PkRzeszPilsu-NO2-1g', 'PkRzeszRejta-O3-1g', 'PkRzeszRejta-SO2-1g']
sensor_column_map = {
    'PkRzeszPilsu-PM2.5-1g': 'PM2.5',
    'PkRzeszPilsu-PM10-1g': 'PM10',
    'PkRzeszPilsu-CO-1g': 'CO',
    'PkRzeszPilsu-NO2-1g': 'NO2',
    'PkRzeszRejta-O3-1g': 'O3',
    'PkRzeszRejta-SO2-1g': 'SO2'
}

last_valid_row = None
for i in range(len(today) - 1, -1, -1):
    row = today.iloc[i]
    if any(not pd.isna(row[col]) for col in sensor_column_map.keys()):
        last_valid_row = row
        break

if last_valid_row is not None:
    max_component, max_caqi, max_value, causes = analyze_caqi_components(last_valid_row[features], tree_model, features, sensor_column_map)
    if max_component:
        print(f"\nNajwiększy wpływ na CAQI ma: {max_component}")
        print(f"CAQI dla tego składnika: {max_caqi:.2f}")
        print(f"Stężenie: {max_value:.2f} µg/m³")
        print("Potencjalne przyczyny:")
        for cause in causes:
            print(f"- {cause}")
else:
    print("Brak dostępnych danych do analizy.")



# Zapis wyników do pliku
with open('caqi_component_analysis.txt', 'w', encoding='utf-8') as f:
    if last_valid_row is not None and max_component:
        f.write(f"Największy wpływ na CAQI ma: {max_component}\n")
        f.write(f"CAQI dla tego składnika: {max_caqi:.2f}\n")
        f.write(f"Stężenie: {max_value:.2f} µg/m³\n")
        f.write("Potencjalne przyczyny:\n")
        for cause in causes:
            f.write(f"- {cause}\n")

    else:
        f.write("Brak dostępnych danych do analizy.\n")