import pandas as pd

def load_accident_data(filepath="data/accident.csv"):
    # accident.csv 파일은 UTF-8 인코딩으로 불러옵니다.
    return pd.read_csv(filepath, encoding="utf-8")

def load_weather_data(filepath="data/weather.csv"):
    # weather.csv 파일은 CP949 인코딩으로 불러옵니다.
    return pd.read_csv(filepath, encoding="cp949")

def load_stat_data(filepath="data/요일별_시간대별_교통사고_20260406161126.csv"):
    # 요일별 통계 파일은 CP949 인코딩으로 불러옵니다.
    return pd.read_csv(filepath, encoding="cp949")