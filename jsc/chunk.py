import os
import logging
from typing import List

from dotenv import load_dotenv

# ✅ LangChain 0.3+ 호환 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

# ✅ Pinecone v3 클라이언트
from pinecone import Pinecone, ServerlessSpec

# ──────────────────────────────────────────────────────────────
# 기본 설정
# ──────────────────────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.getenv("INDEX_NAME", "2pj-index")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")

# bge-m3 임베딩 차원 (로컬 bge-m3 567m은 1024 차원)
EMBED_DIM = 1024
METRIC = "cosine"
NAMESPACE = "default"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


# ──────────────────────────────────────────────────────────────
# 1) 입력 파일 로드 (txt 전용, 인코딩 자동 폴백)
# ──────────────────────────────────────────────────────────────
def load_txt(filepath: str) -> str:
    """
    TXT 파일을 문자열로 로드. utf-8 실패 시 cp949 폴백.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="cp949") as f:
            text = f.read()
    logging.info(f"문서 길이: {len(text)}자")
    return text


# ──────────────────────────────────────────────────────────────
# 2) 문서 길이에 따른 동적 청크 설정 (글자 수 기준)
# ──────────────────────────────────────────────────────────────
def dynamic_split(text: str):
    """
    문서 길이별로 청크 사이즈/오버랩을 동적으로 결정하고 분할.
    """
    length = len(text)

    # 기준은 실제 운영하면서 조정 가능
    # if length < 1000:
    #     chunk_size, chunk_overlap = 200, 20
    # elif length < 3000:
    #     chunk_size, chunk_overlap = 400, 40
    # elif length < 8000:
    #     chunk_size, chunk_overlap = 600, 60
    # else:
    #     chunk_size, chunk_overlap = 800, 80

    if length < 5000:
        chunk_size, chunk_overlap = 1500, 200
    # elif length < 3000:
    #     chunk_size, chunk_overlap = 400, 40
    # elif length < 8000:
    #     chunk_size, chunk_overlap = 600, 60
    else:
        chunk_size, chunk_overlap = 1800, 300

    splitter = RecursiveCharacterTextSplitter(
        # 줄바꿈 > 마침표 > 공백 우선 분리 시도, 그래도 못 나누면 글자단위
        separators=["\n\n", "\n", "。", "!", "?", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,       # 문자 수 기준
        is_separator_regex=False,
    )

    docs = splitter.create_documents([text])
    logging.info(f"총 청크 수: {len(docs)} (청크 크기: {chunk_size}, 오버랩: {chunk_overlap})")
    return docs


# ──────────────────────────────────────────────────────────────
# 3) Pinecone 인덱스 준비 (v3)
# ──────────────────────────────────────────────────────────────
def ensure_pinecone_index(pc: Pinecone, index_name: str):
    """
    인덱스 없으면 생성. bge-m3 기준 dimension=1024, cosine.
    """
    # v3의 list_indexes는 IndexList를 반환 (names() 지원), 호환 처리
    try:
        existing_names = pc.list_indexes().names()
    except Exception:
        existing_names = [x["name"] for x in pc.list_indexes()]  # fallback

    if index_name not in existing_names:
        logging.info(f"인덱스 '{index_name}'가 없어 생성합니다...")
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        logging.info(f"인덱스 '{index_name}' 생성 완료.")
    else:
        logging.info(f"인덱스 '{index_name}' 이미 존재.")


# ──────────────────────────────────────────────────────────────
# 4) 임베딩 & 업로드
# ──────────────────────────────────────────────────────────────
def embed_and_upload(docs: List):
    """
    - Ollama 임베딩으로 문서를 벡터화
    - Pinecone VectorStore로 업로드
    """
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY가 설정되지 않았습니다. .env를 확인하세요.")

    # ✅ Pinecone v3 클라이언트 인스턴스
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 인덱스 보장
    ensure_pinecone_index(pc, INDEX_NAME)

    # ✅ langchain-ollama Embeddings (Deprecation 대응)
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    # ✅ langchain-pinecone VectorStore (LangChain 0.3+)
    # from_documents 내부에서 upsert 수행
    _ = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )

    logging.info("벡터 업로드 완료")


# ──────────────────────────────────────────────────────────────
# 5) 실행 진입점
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 팀 공용 샘플 경로 예시
    file_path = r"guidebook.txt"

    text = load_txt(file_path)
    chunks = dynamic_split(text)
    embed_and_upload(chunks)
