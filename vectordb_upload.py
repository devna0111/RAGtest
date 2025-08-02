from parsing_utils import split_chunks
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def data_to_vectorstore(file_path:str)-> bool :
    # 1. 로컬 QdrantClient 연결
    client = QdrantClient(host="localhost", port=6333) # Qdrant 로컬에서 열린 서버 객체
    print("Qdrant server open")
    # client.delete_collection(file_path) # 컬렉션 삭제
    client.create_collection(
        collection_name="local_docs",
        vectors_config=VectorParams(size=1024, # bge-m3:567m의 벡터 사이즈
                                    distance=Distance.COSINE)
    )

    # 2. 임베딩 모델
    embedding_function = OllamaEmbeddings(model="bge-m3:567m") # vectorsize=1024

    # 4. 문서 준비
    documents = split_chunks(file_path)

    # 5. Qdrant VectorStore 수동 생성
    qdrant = Qdrant(
        client=client,
        collection_name="local_docs",
        embeddings=embedding_function,
    )

    # 6. 문서 삽입
    qdrant.add_documents(documents)
    
    return qdrant


if __name__ == "__main__" :
    file_path = "sample_inputs/sample.png"
    qdrant = data_to_vectorstore(file_path)    
    # 7. 유사도 검색
    query = "이 문서가 어떤 시스템인지 알려줘"
    results = qdrant.similarity_search(query, k=3)
    print(results)
