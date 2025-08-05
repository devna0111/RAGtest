import os
import logging
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# ──────────────────────────────────────────────────────────────
# 1) .env: API Key만 사용
# ──────────────────────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY가 .env에 없습니다.")

# ──────────────────────────────────────────────────────────────
# 2) 고정 기본값 (프로젝트 지침)
# ──────────────────────────────────────────────────────────────
INDEX_NAME = "2pj-index"
NAMESPACE = "default"
PINECONE_REGION = "us-east-1"   # Pinecone v3 serverless spec에 맞춰 생성되어 있어야 함
PINECONE_CLOUD = "aws"

# 모델: 지침값
OLLAMA_EMBED_MODEL = "bge-m3"
OLLAMA_TEXT_MODEL = "gemma3:1b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# ──────────────────────────────────────────────────────────────
# 3) VectorStore & Retriever
# ──────────────────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

# 업로드해 둔 인덱스에 연결
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 5},
)

# ──────────────────────────────────────────────────────────────
# 4) 프롬프트
# ──────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template(
    """당신은 사내 문서 정리 및 설명에 특화된 AI 비서입니다.
다음은 문서에서 검색된 내용입니다:

{context}

위의 내용을 바탕으로 사용자의 질문에 대해
- 구체적인 출처 경로 또는 예시를 제시하고
- 문장 구조를 명확히 하며
- 실무적으로 바로 활용할 수 있도록
한국어로 정확하고 친절하게 답변해주세요.
질문: {question}"""
)

# ──────────────────────────────────────────────────────────────
# 5) LLM & 체인 구성
#    ⚠ 핵심: retriever 앞에 itemgetter("question")로 문자열만 전달
# ──────────────────────────────────────────────────────────────
llm = ChatOllama(model=OLLAMA_TEXT_MODEL)

chain: Runnable = (
    {
        "context": itemgetter("question") | retriever,  # 문자열만 전달
        "question": itemgetter("question"),
    }
    | prompt
    | llm
)

# ──────────────────────────────────────────────────────────────
# 6) 실행 함수
# ──────────────────────────────────────────────────────────────
def ask_question(user_input: str):
    logging.info(f"질문: {user_input}")
    resp = chain.invoke({"question": user_input})
    print("\n답변:\n", resp.content)


if __name__ == "__main__":
    while True:
        q = input("질문을 입력하세요 (종료: exit): ")
        if q.lower() in ("exit", "quit"):
            break 
        ask_question(q)
