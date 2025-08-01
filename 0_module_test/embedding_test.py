from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import time
texts = [
    "사내 업무 자동화를 위한 프로젝트입니다.",
    "우리는 문서를 요약하고 퀴즈를 생성합니다.",
    "발표 영상을 분석하여 피드백을 제공합니다."
]

long_text = ["""
사내 업무 자동화를 위한 시스템은 다양한 문서 유형(doc, pdf, ppt, xlsx, 이미지 등)을 입력받아, 이를 분석하고 요약하는 기능을 제공합니다.
예를 들어, 사용자가 업로드한 파일에서 표나 그래프, 이미지 내 텍스트 등을 인식하여 전체 내용을 요약하고, 해당 내용을 기반으로 보고서나
발표자료를 생성할 수 있습니다. 이 과정에서 LangChain과 Ollama의 텍스트 및 비전 모델이 함께 사용되며, 분석된 데이터는 벡터 DB에 비동기적으로
업로드됩니다. 이렇게 저장된 벡터들은 사용자의 질의에 응답할 수 있도록 검색 인덱스로 활용되며, 유사도 기반 검색을 통해 가장 관련성 높은 정보를
추출할 수 있습니다. 또한 사용자의 요청에 따라 입력된 문서 내용으로부터 퀴즈를 생성하거나, 사전 정의된 템플릿 기반 보고서를 자동으로 구성할 수 있으며,
발표 영상에서 음성을 추출하고 피치, 억양, 발화속도, 음량 등을 분석하여 비언어적 표현에 대한 피드백까지 제공합니다. 이 시스템은 CUDA 11.2 및 Python 3.10을 기반으로
로컬 또는 폐쇄망 환경에서 작동하며, bge-m3:567m 임베딩 모델을 통해 문맥 정보를 효율적으로 벡터화하고, qwen2.5vl:7b 모델을 활용해 이미지 및 멀티모달
입력을 처리합니다. 사용자는 웹 기반 인터페이스를 통해 문서를 업로드하고 결과를 확인할 수 있으며, Flask를 통한 API 연동으로 자동화된 파이프라인 구성이
가능합니다. 예를 들어, 사내 교육 자료를 분석하여 퀴즈를 자동으로 생성하거나, 업무 매뉴얼을 요약하여 발표자료로 변환하는 등의 다양한 활용이 가능합니다.
또한, 발표 영상을 분석해 발표자의 시선 처리, 표정, 제스처와 같은 비언어적 요소를 LangChain + MediaPipe + Whisper 등을 조합하여 실시간 또는 배치 분석
방식으로 피드백할 수 있습니다. 이처럼 통합된 문서 분석 및 피드백 시스템은 사내 교육, 품질 관리, 보고서 자동화 등 다양한 업무 효율 향상에 기여할 수 있습니다.
"""]

embedding = OllamaEmbeddings(model="bge-m3:567m") # vectorsize : 1024
start = time.time()
# 벡터 DB (Chroma) 생성 및 임베딩 저장 
vectorstore = Chroma.from_texts(long_text, embedding=embedding, collection_name="search_demo")
mid = time.time()

query = "발표 피드백 기능이 있나요?"

# 유사도 기반 검색
results = vectorstore.similarity_search_with_score(query, k=2)
end = time.time()
# for i, doc in enumerate(results, 1):
#     print(f"[Top {i}] {doc[0].page_content} : {doc[1]}") # Chroma에선 score가 낮을 수록 유사
print(results)
print(len(long_text))
print("임베딩 시간", mid - start)
print("검색 시간", end - mid)
print("전체 시간", end - start)