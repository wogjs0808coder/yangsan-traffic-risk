import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .data_loader import load_accident_data, load_weather_data


def parse_year_month(text):
    # 발생년월 컬럼에서 연도와 월을 추출하는 정규식 함수입니다.
    if pd.isna(text):
        return None, None
    match = re.search(r"(\d+)년\s*(\d+)월", str(text))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def preprocess_and_save(output_dir="data_processed/"):
    # 1. 원본 데이터 로드
    acc = load_accident_data()
    weather = load_weather_data()

    # 2. 날씨 데이터 전처리
    # 강수량 결측치는 0.0으로 대체합니다.
    weather["일강수량(mm)"] = weather["일강수량(mm)"].fillna(0.0)

    # 평균기온 및 상대습도의 결측치는 선형 보간 후 전후값으로 채웁니다.
    num_cols = ["평균기온(°C)", "평균 풍속(m/s)", "평균 상대습도(%)"]
    weather[num_cols] = (
        weather[num_cols].interpolate(method="linear").ffill().bfill()
    )

    # 날짜 연월 데이터 생성 및 정수형(int)으로 강제 변환
    weather["일시"] = pd.to_datetime(weather["일시"])
    weather["year"] = weather["일시"].dt.year.astype(int)
    weather["month"] = weather["일시"].dt.month.astype(int)

    # 트러블슈팅: 폭우 시 교통량 감소로 인한 사고 건수 왜곡(이상치 편향) 방지
    # 강수량을 최대 100mm로 클리핑하여 데이터의 편향 학습을 방지합니다.
    weather["일강수량_클립(mm)"] = np.clip(weather["일강수량(mm)"], 0.0, 100.0)

    # 100mm를 초과하는 날은 재난 상황 범주 플래그를 별도로 신설합니다.
    weather["폭우_재난_플래그"] = (weather["일강수량(mm)"] > 100.0).astype(int)

    # 일별 데이터를 월 단위 통계값으로 축소 집계(Aggregation)합니다.
    monthly_weather = (
        weather.groupby(["year", "month"])
        .agg(
            {
                "평균기온(°C)": "mean",
                "일강수량_클립(mm)": "sum",
                "평균 풍속(m/s)": "mean",
                "평균 상대습도(%)": "mean",
                "폭우_재난_플래그": "max",
            }
        )
        .reset_index()
    )

    # 3. 사고 데이터 날짜 파싱 (연도와 월을 각각 올바르게 분리)
    parsed_results = acc["발생년월"].apply(parse_year_month)
    acc["year"] = [res[0] for res in parsed_results]
    acc["month"] = [res[1] for res in parsed_results]

    # 병합 전 키값 타입 명확히 정수형으로 통일
    acc["year"] = acc["year"].astype(int)
    acc["month"] = acc["month"].astype(int)
    monthly_weather["year"] = monthly_weather["year"].astype(int)
    monthly_weather["month"] = monthly_weather["month"].astype(int)

    # 4. 이종 데이터 시간대 정합 조인 (Daily-Monthly Join)
    df = pd.merge(acc, monthly_weather, on=["year", "month"], how="left")

    # 5. 개인화 피처 선택 및 인코딩
    cat_features = [
        "주야",
        "기상상태",
        "노면상태",
        "가해운전자 차종",
        "가해운전자 연령대",
    ]
    num_features = [
        "평균기온(°C)",
        "일강수량_클립(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "폭우_재난_플래그",
    ]

    # 범주형 피처 원핫 인코딩
    X_cat = pd.get_dummies(df[cat_features], drop_first=True)
    X_num = df[num_features]
    X = pd.concat([X_num, X_cat], axis=1)

    # 타겟 변수(사고유형) 수치 인코딩
    le = LabelEncoder()
    y = le.fit_transform(df["사고유형"])

    # 6. 학습용/평가용 데이터 분할 (80 대 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 전처리 완료 데이터 보관 폴더 생성 및 저장
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(
        os.path.join(output_dir, "y_train.csv"), index=False
    )
    pd.DataFrame(y_test, columns=["target"]).to_csv(
        os.path.join(output_dir, "y_test.csv"), index=False
    )

    # 모델 평가 시 텍스트 클래스를 출력하기 위해 매핑 정보 저장
    pd.DataFrame(le.classes_, columns=["class_name"]).to_csv(
        os.path.join(output_dir, "classes.csv"), index=False
    )

    print("전처리가 성공적으로 완료되었습니다. 저장 경로:", output_dir)


if __name__ == "__main__":
    preprocess_and_save()