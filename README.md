# yangsan-traffic-risk
# 🚦 양산시 교통사고 예측 및 안전 가이드 프로젝트

본 프로젝트는 양산시의 교통사고 이력과 기상 데이터를 활용하여, 운전자 프로필 및 기상 조건에 따른 사고 위험도를 예측하고 데이터 무결성을 확보하는 파이프라인을 구축한 프로젝트입니다.

## 📝 상세 포트폴리오 및 트러블슈팅
본 프로젝트의 상세한 기획 배경, 데이터 정합성 해결 과정, 그리고 900mm 강수량 이상치 트러블슈팅 등의 전체 과정은 아래 노션 블로그에 정리되어 있습니다.
👉 [Notion 기술 블로그 바로가기]((https://fragrant-dewberry-Za3.notion.site/3c0294bdb6c580c394e1dfc48ae705a6?source=copy_link))

---

## 🏗 Directory Architecture
```text
yangsan-traffic-risk/
├── data/
│   ├── accident.csv          # 양산시 사고 이력 데이터 (cp949)
│   └── weather.csv           # 양산시 기상 데이터 (cp949)
├── src/
│   ├── data_loader.py        # 인코딩 처리 및 데이터 로드
│   ├── preprocessor.py       # 이상치 제거 및 결측치 선형 보간
│   └── model_trainer.py      # XGBoost 기반 모델 학습 및 검증
└── README.md
