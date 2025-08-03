# llm_utils/quiz_generator.py
from langchain_ollama import ChatOllama

def generate_quiz_from_text(text: str, num_questions: int = 10) -> str:
    """
    문서 내용을 기반으로 이해도 체크를 위한 객관식 문제를 생성합니다.
    """
    prompt = f"""
당신은 교육용 문서를 기반으로 퀴즈를 생성하는 전문가입니다.
다음 문서를 바탕으로 이해도를 확인할 수 있는 객관식 문제 {num_questions}개를 만들어주세요.
각 문항은 보기 5개와 정답 표시를 포함해야 합니다.

[문서 내용]
{text}
"""
    llm = ChatOllama(model="qwen2.5vl:7b")
    return llm.invoke(prompt).content

if __name__ == "__main__" :
    text="""서울 PM 재배치를 위한 수요 예측 모델 개발

프로젝트 명	서울 PM 재배치를 위한 수요 예측
참여 인원	4명
기간	2025.07.10 ~ 2025.07.23
개발 환경	Language	Python 3.10
	Framework	Flask (API), Jinja2 (Template)
	IDE Tool	VSCode, Jupyter Notebook, WorkBench
	DB	MySQL
	Open Source		TensorFlow, scikit-learn, lightgbm, pandas, numpy, matplotlib, seaborn, seleninum, 공공데이터포털 Open API(data.go.kr), OpenAI_API, Prophet, openmeteoAPI
	데이터 출처	서울시 월별 공공자전거 대여 – 서울시 열린데이터광장
기상 및 기후 – 기상청 통계 포털
PM 견인률 및 운영 현황 – 공공데이터 포털
	프로젝트 기여	1. 데이터 분석(인구, 인프라, 시계열 속성)
가.	인구 데이터의 경우 3종의 데이터를 확인(행정인구, 생활인구, 유동인구) : 데이터가 갖는 의미에 대해 분석하고 프로젝트의 적합성(실제 PM을 이용할 인구) 체크
나.	인프라 데이터의 경우 자전거 노선의 길이와 그 수에 따라 대여량과의 상관관계를 확인하고 중요도를 체크, Feature 사용여부 결정
다.	외부 활용 자원에 따라 계절성, FM/LM의 활용에 따른 시간에 따른 영향 체크를 통해 시계열 속성 체크
2. 데이터 전처리
가. 데이터의 결측치를 확인 
: 공공 제공 데이터이며 5분 간격의 데이터: 결측치 0건
나. 정류소 중 일반인이 사용 가능한 정류소 + 수리 등을 위한 정비정류소를 확인, 프로젝트 목적에 적합한 일반인 사용 정류소만 추출
다. 데이터 이상치 처리 : 수요가 많을 것으로 예상되는 데이터는 이상치로 선 처리가 아닌 모델 체크 후처리로 대응
3. 모델의 설계
가. 필요 모델 체크 : 최초 가설 설정 시 민간 PM의 경우 공공자전거의 대여량 영향을 설정, 이에 따라 필요 모델 최소 2개 분류(공공자전거 대여량 예측 / 민간 PM 대여량 예측)
나. 공공자전거 대여량 예측 모델
- 시계열 속성과 타겟 변수(대여량)의 불균등 분포, 극단적 값에 따라 모델을 설계
- 최초 Classification 계획하였으나 타겟 변수 분포에 따라 PM수요 예측 모델에 부적합 판단, Linear Regression 모델로 변경
- LSTM(중장기 데이터 예측에 강점), DNN(시계열 속성을 작년도 대여량을 Feature로 활용하여 해소), 앙상블 모델(LightGBM(작년도 대여량 Feature의 활용에 따라 복잡한 Feature 분석에 강한 모델 선정))
다.	PM 대여량 예측 모델
- 공공자전거 대여량 예측 모델의 결과를 Feature 활용하는 모델의 설계
- 공공자전거 대여량 예측 모델과 달리 재배치가 필요한 시간대를 예측함에 큰 목표로, 사분위수 Q3 초과에 대해 재배치가 가능하게 Classification 모델 설계
4. 웹서비스 구현
가. 웹서비스가 가능하게끔 기능들의 함수화 진행
- 전년도 데이터 MySQL load
- 실시간 기상 예보(openmeteoAPI)
- 공공자전거 대여량 예측 모델 예측
- 민간 PM 대여량 예측
- 행정구별 주요 역들에 대해 대여량 가중치에 따라 
상위 5개 역에 대한 대여량 예측
- 시각화 함수 등
나. flask와 Jinja2 템플릿을 통한 웹서비스 내용 구현
다. openAI API를 활용한 재배치 필요 시간대 확인과 데이터 분석 기능 구현
	협업도구	GitHub, Google Drive, Notion
"""
    print(generate_quiz_from_text(text))