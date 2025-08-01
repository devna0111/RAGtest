from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document

# 1. 로컬 QdrantClient 연결
client = QdrantClient(host="localhost", port=6333) # Qdrant 로컬에서 열린 서버 객체
client.delete_collection("local_docs") # 컬렉션 삭제
client.create_collection(
    collection_name="local_docs",
    vectors_config=VectorParams(size=1024, # bge-m3:567m의 벡터 사이즈
                                distance=Distance.COSINE)
)

# 2. 임베딩 모델
embedding_function = OllamaEmbeddings(model="bge-m3:567m") # vectorsize=1024

# 4. 문서 준비
documents = [
    Document(page_content="사내 문서 자동화를 위해 다양한 포맷의 문서를 처리할 수 있는 시스템을 구축하고 있습니다."),
    Document(page_content="이 시스템은 사내 교육 자료를 자동으로 요약하고, 퀴즈까지 생성할 수 있어 학습 효과를 높입니다."),
    Document(page_content="Flask 기반의 API 서버를 통해 사용자 인터페이스와 자동화 백엔드를 연동합니다."),
    Document(page_content="업로드된 PDF, PPT, Excel 문서는 LangChain과 Ollama를 이용해 분석 후 벡터화됩니다."),
    Document(page_content="문서 속 표와 그래프를 추출한 후 정확한 수치를 기반으로 보고서를 생성합니다."),
    Document(page_content="발표 영상의 음성을 추출하여 Whisper로 STT 처리한 후, 내용 요약 및 말하기 속도, 억양 등을 분석합니다."),
    Document(page_content="MediaPipe를 통해 사용자의 제스처, 시선, 표정을 분석하여 비언어적 발표 스킬 피드백을 제공합니다."),
    Document(page_content="벡터 DB는 Qdrant를 사용하여 문서 간 유사도 검색을 빠르게 수행할 수 있습니다."),
    Document(page_content="사전 정의된 템플릿을 활용해 보고서와 발표안을 자동 생성하는 기능도 포함되어 있습니다."),
    Document(page_content="CUDA 가속 환경에서 Ollama의 bge-m3 모델을 활용하여 1024차원 임베딩을 생성합니다."),
    Document(page_content="""
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
""")
]

# 5. Qdrant VectorStore 수동 생성
qdrant = Qdrant(
    client=client,
    collection_name="local_docs",
    embeddings=embedding_function,
)

# 6. 문서 삽입
qdrant.add_documents(documents)

# 7. 유사도 검색
query = "이 문서가 어떤 시스템인지 알려줘"
results = qdrant.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results):
    print(f"[Top {i+1}] Score: {score:.3f}")
    print(doc.page_content)
