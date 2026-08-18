import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate(data_dir="data_processed/"):
    # 1. 전처리된 데이터 로드
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["target"]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["target"]
    classes = pd.read_csv(os.path.join(data_dir, "classes.csv"))["class_name"].tolist()
    
    # 2. XGBoost 다중 분류기 모델 정의 및 학습
    print("양산시 교통사고 위험도 예측을 위한 XGBoost 모델을 학습합니다...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss"
    )
    
    model.fit(X_train, y_train)
    
    # 3. 모델 성능 테스트 및 평가 보고서 도출
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print("모델 평가 결과:")
    print("테스트 정확도(Accuracy):", round(acc, 4))
    print("\n상세 검증 성능 보고서 (Classification Report):")
    print(classification_report(y_test, predictions, target_names=classes, zero_division=0))
    
    # 4. 프로덕션 서빙용 학습 모델 파일 내보내기
    model_path = os.path.join(data_dir, "xgboost_traffic_model.json")
    model.save_model(model_path)
    print("학습된 XGBoost 모델이 안전하게 보관되었습니다. 파일 위치:", model_path)

if __name__ == "__main__":
    train_and_evaluate()