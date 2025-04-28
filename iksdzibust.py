import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

# Wczytanie danych z pliku Excel
data = pd.read_excel("2023 dane.xlsx")

# Zamiana kolumny 'Date' na datę
data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y %H:%M')

# Ekstrakcja cech czasowych (możesz je pominąć jeśli nie są potrzebne)
data['hour'] = data['Date'].dt.hour
data['day_of_week'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month

# Zastąpienie brakujących wartości metodą interpolacji
data.fillna(data.mean(), inplace=True)


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

# Funkcja do obliczenia CAQI
def compute_caqi_row(row):
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
data['CAQI'] = data.apply(compute_caqi_row, axis=1)

# Usunięcie wierszy z brakiem wartości CAQI (jeśli jakieś istnieją)
data = data.dropna(subset=['CAQI'])

# Zakładając, że cechy to zmienne zanieczyszczeń
features = ['PkRzeszPilsu-PM2.5-1g', 'PkRzeszPilsu-PM10-1g', 'PkRzeszPilsu-CO-1g',
            'PkRzeszPilsu-NO2-1g', 'PkRzeszRejta-O3-1g', 'PkRzeszRejta-SO2-1g']

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