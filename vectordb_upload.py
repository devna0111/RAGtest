import os
import hashlib
from parsing_utils import split_chunks
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def get_file_hash(file_path: str) -> str:
    """파일의 해시값을 기반으로 고유 컬렉션 이름 생성"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def data_to_vectorstore(file_path: str):
    # Qdrant 클라이언트 연결
    client = QdrantClient(host="localhost", port=6333)
    print("[Qdrant 연결 성공]")

    # 파일 기반 고유 컬렉션 이름 생성
    collection_name = f"doc_{get_file_hash(file_path)}"

    # 이미 존재하는 컬렉션인지 확인
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name in existing_collections:
        print(f"[ℹ이미 존재하는 컬렉션: {collection_name}] → 기존 DB 사용")
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=OllamaEmbeddings(model="bge-m3:567m")
        )

    # 컬렉션 새로 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"[새 컬렉션 생성: {collection_name}]")

    # 문서 임베딩
    documents = split_chunks(file_path)
    embedding_function = OllamaEmbeddings(model="bge-m3:567m")
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_function
    )
    qdrant.add_documents(documents)
    print(f"[문서 {len(documents)}개 벡터 저장 완료]")

    return qdrant


if __name__ == "__main__":
    file_path = "sample_inputs/sample.txt"
    qdrant = data_to_vectorstore(file_path)

    query = "이 문서로 발표안을 짜줘"
    results = qdrant.similarity_search(query, k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n[Top {i}]")
        print(doc.page_content[:100])
