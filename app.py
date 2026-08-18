import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# 페이지 기본 설정 및 디자인
st.set_page_config(
    page_title="양산시 교통사고 위험도 예측 시스템",
    page_icon="🚦",
    layout="wide"
)

# 데이터 및 모델 로드 함수 (캐싱을 통해 속도 향상)
@st.cache_resource
def load_prediction_resources():
    model_path = "data_processed/xgboost_traffic_model.json"
    classes_path = "data_processed/classes.csv"
    train_cols_path = "data_processed/X_train.csv"
    
    # 필수 파일 존재 여부 확인
    if not (os.path.exists(model_path) and os.path.exists(classes_path) and os.path.exists(train_cols_path)):
        return None, None, None

    # 모델 객체 생성 및 로드
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 클래스명 및 학습 피처 열 리스트 로드
    classes = pd.read_csv(classes_path)["class_name"].tolist()
    train_cols = pd.read_csv(train_cols_path).columns.tolist()
    
    return model, classes, train_cols

model, classes, train_cols = load_prediction_resources()

# 메인 헤더 영역
st.title("🚦 양산시 개인화 교통사고 위험도 예측 시스템 (v1)")
st.write("실시간 기상 기후 조건과 운전자의 차량/연령 설정을 기반으로 예측 엔진이 판단한 최우선 위험 시나리오와 맞춤형 법규 준수 가이드를 제시합니다.")

if model is None:
    st.error("데이터 폴더에 학습된 모델 정보나 전처리 파일이 존재하지 않습니다. 먼저 src/preprocessor.py와 src/model_trainer.py를 실행하여 모델 학습을 완료해 주세요.")
else:
    # [수정 1] 화면 레이아웃 분할 (왼쪽: 사용자 입력 폼, 오른쪽: 결과 분석 리포트)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠 실시간 운행 환경 컨텍스트 설정")
        
        # 기상 및 환경 정보 입력
        st.write("기상 및 노면 상태")
        temp = st.slider("현재 기온 (°C)", -15.0, 40.0, 15.0)
        rain = st.number_input("현재 일강수량 (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        wind = st.slider("평균 풍속 (m/s)", 0.0, 15.0, 2.0)
        humidity = st.slider("평균 상대습도 (%)", 10, 100, 60)
        
        juya = st.selectbox("시간대 설정", ["주간", "야간"])
        weather_state = st.selectbox("하늘 기상 상태", ["맑음", "비", "흐림", "기타"])
        road_state = st.selectbox("도로 노면 상태", ["건조", "젖음/습기", "침수", "기타"])
        
        # 운전자 프로필 정보 입력
        st.write("가해 운전자 조건 설정")
        car_type = st.selectbox("차량 종류", ["승용", "화물", "이륜", "승합", "자전거"])
        age_group = st.selectbox("운전자 연령대", ["21-30세", "31-40세", "41-50세", "51-60세", "65세 이상", "20세 이하"])

    with col2:
        st.subheader("📊 인공지능 실시간 안전 리포트")
        
        # 트러블슈팅: 100mm를 초과하는 극한 폭우 상황 감지 및 처리 알림
        if rain >= 100.0:
            st.warning("⚠️ 긴급 재난 상황 경보: 강수량이 100mm를 초과하는 재난 수준의 폭우가 감지되었습니다. 차량 침수 방지 및 도로 통제 상황을 확인하시고 외출을 엄격히 자제해 주세요. (본 시스템은 극한 폭우 시의 통계적 왜곡을 차단하기 위해 이상치 클리핑 기법을 적용하여 정상 위험도를 산출합니다.)")
        
        # 1. 모델 예측용 입력 데이터프레임 빌드 (One-Hot Encoding 대응)
        # 학습에 사용된 모든 피처를 0으로 채운 한 행의 데이터프레임 생성
        input_data = pd.DataFrame(0, index=[0], columns=train_cols)
        
        # 수치형 변수 대입
        input_data["평균기온(°C)"] = temp
        input_data["일강수량_클립(mm)"] = min(rain, 100.0)  # 이상치 클리핑 로직 적용
        input_data["평균 풍속(m/s)"] = wind
        input_data["평균 상대습도(%)"] = humidity
        input_data["폭우_재난_플래그"] = 1 if rain > 100.0 else 0
        
        # 범주형 원핫 인코딩 대입 (drop_first=True 조건 맞춤형 설계)
        # 선택한 카테고리 값이 원핫 컬럼 이름에 존재할 경우에만 1을 할당
        for cat_col, selected_val in [
            ("주야", juya), 
            ("기상상태", weather_state), 
            ("노면상태", road_state), 
            ("가해운전자 차종", car_type), 
            ("가해운전자 연령대", age_group)
        ]:
            col_name = f"{cat_col}_{selected_val}"
            if col_name in input_data.columns:
                input_data[col_name] = 1
                
        # 2. 실시간 AI 추론 및 확률 예측
        # [수정 2] predict_proba는 2차원 배열을 반환하므로 [0]을 붙여 첫 번째 데이터의 확률값 리스트만 추출
        pred_probs = model.predict_proba(input_data)[0]
        
        # 예측된 사고유형별 확률을 데이터프레임으로 정리
        result_df = pd.DataFrame({
            "사고 유형": classes,
            "발생 위험도 (%)": [round(prob * 100, 2) for prob in pred_probs]
        }).sort_values(by="발생 위험도 (%)", ascending=False)
        
        # [수정 3] 가장 위험한 최고 순위 사고 유형 및 수치 파악 (iloc 뒤에 [0] 추가)
        top_risk = result_df.iloc[0]
        top_accident = top_risk["사고 유형"]
        top_percentage = top_risk["발생 위험도 (%)"]
        
        # 위험 등급 색상 분류 및 알림 카드 출력
        if top_percentage >= 50.0:
            status_color = "red"
            status_text = "매우 위험"
        elif top_percentage >= 35.0:
            status_color = "orange"
            status_text = "경계"
        else:
            status_color = "green"
            status_text = "보통"
            
        st.markdown(f"#### 현재 종합 위험등급: <span style='color:{status_color}; font-weight:bold;'>{status_text} ({top_percentage}%)</span>", unsafe_allow_html=True)
        st.write(f"현재 조건 하에서 통계적으로 발생 확률이 가장 높게 분석된 사고 형태는 **[{top_accident}]** 입니다.")
        
        # 사고 유형별 지능형 안전 가이드 브리핑 (도메인 지식 융합 규칙)
        st.markdown("---")
        st.markdown("#### 💡 맞춤형 안전 운전 가이드")
        
        if "추돌" in str(top_accident):
            st.info("안전거리 미확보로 인한 충돌 위험이 높습니다. 평소 대비 2배 이상의 차간 거리를 준수하시고 크루즈 컨트롤 시스템 속도를 낮게 조정하세요.")
        elif "충돌" in str(top_accident):
            st.info("교차로 꼬리물기 및 신호 위반 패턴에서 자주 유발되는 사고 유형입니다. 황색 신호 시 무리하게 진입하지 마시고 좌우 시야를 선제적으로 확보하세요.")
        elif "사람" in str(top_accident):
            st.warning("보행자 사고 발생률이 지배적으로 도출되었습니다. 스쿨존 및 횡단보도 접근 시 주행 속도를 30km/h 미만으로 감속하시고 사각지대를 방어 운전하세요.")
        elif "단독" in str(top_accident):
            st.warning("노면 결빙 및 젖은 도로에서의 수막현상으로 차선 이탈이나 차량 전복 위험이 포착되었습니다. 급제동 및 급가속을 전면 금지해 주세요.")
        else:
            st.write("주변 기상 상황과 교통 흐름을 예의주시하여 전방 태만을 최소화해 주세요.")
            
        # 3. 사고유형별 전체 위험 확률 분포 차트 시각화
        st.markdown("---")
        st.markdown("#### 📊 사고유형별 통계 예측 분포")
        
        # 인덱스 리셋 후 차트 생성
        st.bar_chart(data=result_df.set_index("사고 유형"))