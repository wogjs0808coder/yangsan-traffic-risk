# yangsan-traffic-risk
# 🚦 양산시 교통사고 예측 및 안전 가이드 프로젝트

> 기상 시계열 데이터와 운전자 프로필을 융합한 개인화 위험도 예측 파이프라인 구축

본 프로젝트는 양산시의 실제 교통사고 이력(`accident.csv`)과 기상 데이터(`weather.csv`)를 결합하여, 실시간 기상 상태와 운전자 속성(차종, 연령대)에 따른 사고 위험도 및 위반 유발 원인을 예측하는 머신러닝 시스템입니다. 단순 위험 예측을 넘어 AI 모델 학습 과정에서의 기상 데이터 편향(Data Bias)을 도메인 지식으로 Troubleshooting한 과정을 담고 있습니다.

---
## 📝 상세 포트폴리오 및 트러블슈팅
본 프로젝트의 상세한 기획 배경, 데이터 정합성 해결 과정, 그리고 900mm 강수량 이상치 트러블슈팅 등의 전체 과정은 아래 노션 블로그에 정리되어 있습니다.
👉 [Notion 기술 블로그 바로가기](https://fragrant-dewberry-0a3.notion.site/3c0294bdb6c580c394e1dfc48ae705a6?source=copy_link)

실시간 AI 대시보드 바로가기
본 프로젝트의 XGBoost 예측 엔진을 활용한 실시간 위험도 시뮬레이션 대시보드가 인터넷상에 배포되어 있습니다. 아래 링크를 통해 직접 테스트해 볼 수 있습니다.
👉 [양산시 교통사고 위험도 예측 대시보드 바로가기](https://yangsan-traffic-risk-xgytxkbwe9a6bkspeaemb9.streamlit.app/)
---

## 🏗 Directory Architecture

```text
yangsan-traffic-risk/
├── data/
│   ├── accident.csv          # 양산시 사고 이력 데이터 (UTF-8 인코딩)
│   ├── weather.csv           # 양산시 일별 기상 데이터 (CP949 인코딩)
│   └── 요일별_시간대별_교통사고.csv # 요일/시간대별 통계 템플릿 (CP949 인코딩)
├── notebooks/
│   └── 01_outlier_analysis.ipynb # 900mm 이상 강수량 모델 편향 원인 분석 및 EDA
├── src/
│   ├── data_loader.py        # 파일별 이종 인코딩 처리 및 메모리 최적화 로더
│   ├── preprocessor.py       # 시계열 정합성 일치(Daily-Monthly Join) 및 강수량 이상치 클리핑
│   └── model_trainer.py      # XGBoost / LightGBM 기반 다중 분류(Multi-class) 모델 학습
├── requirements.txt          # 개발 환경 의존성 (pandas, scikit-learn, xgboost)
└── README.md
