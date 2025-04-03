import requests
import datetime
import pandas as pd

'''
PkRzeszPilsu-PM2.5-1 - https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/20277
PkRzeszPilsu-PM10-1g - https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16344
PkRzeszPilsu-CO-1g - https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16343
PkRzeszPilsu-NO2-1g - https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16319
PkRzeszRejta-O3-1g - https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/4385
PkRzeszRejta-SO2-1 - https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/4391
'''

apiLinks = ["https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/20277", "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16344",
            "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16343", "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/16319",
            "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/4385", "https://api.gios.gov.pl/pjp-api/v1/rest/data/getData/4391"]

def getTodaysData(apiLinks):
    todayData = {}
    for sensor in apiLinks:
        dateList = []
        valueList = []
        response = requests.get(sensor).json()
        data = response['Lista danych pomiarowych']
        for i in range(len(data)):
            if pd.to_datetime(data[i]['Data']).day == datetime.datetime.now().day:
                dateList.append(data[i]['Data'])
                valueList.append(data[i]['Wartość'])

            todayData.update({data[i]['Kod stanowiska']: valueList})
    todayData.update({"Date": dateList})

    return pd.DataFrame(todayData)


getTodaysData(apiLinks).to_csv('out.csv', index=False)