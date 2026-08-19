# yangsan-traffic-risk
# 🚦 양산시 교통사고 예측 및 안전 가이드 프로젝트

> 기상 시계열 데이터와 운전자 프로필을 융합한 개인화 위험도 예측 파이프라인 구축

본 프로젝트는 양산시의 실제 교통사고 이력(`accident.csv`)과 기상 데이터(`weather.csv`)를 결합하여, 실시간 기상 상태와 운전자 속성(차종, 연령대)에 따른 사고 위험도 및 위반 유발 원인을 예측하는 머신러닝 시스템입니다. 단순 위험 예측을 넘어 AI 모델 학습 과정에서의 기상 데이터 편향(Data Bias)을 도메인 지식으로 Troubleshooting한 과정을 담고 있습니다.

---
## 📝 상세 포트폴리오 및 트러블슈팅
본 프로젝트의 상세한 기획 배경, 데이터 정합성 해결 과정, 그리고 900mm 강수량 이상치 트러블슈팅 등의 전체 과정은 아래 노션 블로그에 정리되어 있습니다.
👉 [Notion 기술 블로그 바로가기](https://fragrant-dewberry-0a3.notion.site/3c0294bdb6c580c394e1dfc48ae705a6?source=copy_link)

## 실시간 AI 대시보드 바로가기
본 프로젝트의 XGBoost 예측 엔진을 활용한 실시간 위험도 시뮬레이션 대시보드가 인터넷상에 배포되어 있습니다. 아래 링크를 통해 직접 테스트해 볼 수 있습니다.
👉 [양산시 교통사고 위험도 예측 대시보드 바로가기](https://yangsan-traffic-risk-gheuv599appgwkiqv4mxbn6.streamlit.app/)
---

## 🏗 Directory Architecture

```text
yangsan-traffic-risk/
├── data/                       # 원본 데이터 보관함 (변경 금지)
│   ├── accident.csv            # 양산시 사고 이력 (발생년월 기준)
│   └── weather.csv             # 양산시 기상 데이터 (일별 기준)
│
├── data_processed/                          # 모델 학습용 정제 데이터 및 산출물
│   ├── X_train.csv, X_test.csv           # 분할된 피처 데이터 (80:20)
│   ├── y_train.csv, y_test.csv            # 분할된 Target 데이터
│   ├── classes.csv                               # LabelEncoder 클래스 매핑 정보
│   └── xgboost_traffic_model.json # 최종 학습 완료된 모델 가중치
│
├── src/                                     # 핵심 파이프라인 순서대로
│   ├── data_loader.py           # [1단계] cp949 인코딩 처리 및 데이터 로드
│   ├── preprocessor.py        # [2단계] 결측치/이상치 처리, 시계열 병합(Join)
│   └── model_trainer.py        # [3단계] XGBoost 학습 및 json 파일 저장
│
├── app.py                              # Streamlit 웹 대시보드 구동 스크립트
├── requirements.txt            # Streamlit Cloud 배포용 패키지 목록
├── README.md                   
└── .gitignore                         # venv/, data/ 등 업로드 제외 목록
