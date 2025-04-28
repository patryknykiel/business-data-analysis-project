import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib

# Funkcje do obliczania CAQI
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

# Funkcja przypisująca opis jakości powietrza
def caqi_description(caqi_value):
    if pd.isna(caqi_value):
        return np.nan
    elif caqi_value <= 25:
        return 'Bardzo dobry'
    elif caqi_value <= 50:
        return 'Dobry'
    elif caqi_value <= 75:
        return 'Średni'
    elif caqi_value <= 100:
        return 'Zły'
    else:
        return 'Bardzo zły'

# Wczytanie danych
data = pd.read_excel('2023 dane.xlsx')

# Lista cech
features = ['PkRzeszPilsu-PM2.5-1g', 'PkRzeszPilsu-PM10-1g',
            'PkRzeszPilsu-CO-1g', 'PkRzeszPilsu-NO2-1g',
            'PkRzeszRejta-O3-1g', 'PkRzeszRejta-SO2-1g']

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

# Model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X, y)

# Zapis modelu
joblib.dump(tree_model, 'decision_tree_regressor.pkl')
print("Model drzewa decyzyjnego został zapisany.")

# Funkcja przewidująca
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