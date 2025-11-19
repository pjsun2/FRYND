# FRYND

Streamlit + LangChain + Google Gemini 2.5 Flash를 활용한 항공권 상담 챗봇입니다. 인천국제공항(ICN), 김포국제공항(GMP), 도쿄 하네다(HND), 나리타(NRT) 사이의 편도 여정만을 대상으로 하며, Amadeus API를 함수 호출(Function Calling) 도구로 연결해 최신 운임을 조회합니다.

## 환경 변수

Amadeus API 자격증명을 아래 환경 변수로 설정해야 합니다. 필요 시 `AMADEUS_HOSTNAME`에 `test`(기본값) 또는 `production`을 지정할 수 있습니다.

```bash
export AMADEUS_CLIENT_ID="<your_amadeus_client_id>"
export AMADEUS_CLIENT_SECRET="<your_amadeus_client_secret>"
# export AMADEUS_HOSTNAME="production"  # 실서비스 사용 시

export GOOGLE_API_KEY="<your_google_gemini_api_key>"
```

## 실행 방법

```bash
# 의존성 설치
pip install -e .

# Streamlit 앱 실행
streamlit run main.py
```

앱이 실행되면 챗 입력창에 원하는 여정을 자연어로 입력하면 됩니다. 챗봇은 LangChain 기반 에이전트로 동작하며 필요한 경우 자동으로 Amadeus 함수 호출 도구를 실행해 결과를 데이터 프레임 표 형태로 보여줍니다.
