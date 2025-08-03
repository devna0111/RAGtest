import os
import hashlib
from parsing_utils import split_chunks
from langchain_qdrant import Qdrant
from langchain_ollama import OllamaEmbeddings, ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from llm_utils.quiz_generator import generate_quiz_from_text
from llm_utils.report_generator import generate_report_from_text
from llm_utils.summary_generator import generate_summary_from_text
from llm_utils.presentation_generator import generate_presentation_from_text

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
    print("bge-m3 준비 완료")
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding_function
    )
    print("임베딩 하고 DB 적재")
    qdrant.add_documents(documents)
    print(f"[문서 {len(documents)}개 벡터 저장 완료]")

    return qdrant

def question_answer_based_vectorstore(file_path, query: str = "문서를 요약해주세요") -> str:
    vector_store = data_to_vectorstore(file_path)

    # 필터 및 함수 매핑
    filter_map = {
        "요약": generate_summary_from_text,
        "퀴즈": generate_quiz_from_text,
        "보고서": generate_report_from_text,
        "발표": generate_presentation_from_text,  # 필요 시 추가
    }

    # 키워드에 해당하는 기능 선택
    for keyword, func in filter_map.items():
        if keyword in query:
            print(f"[필터 적용: type={keyword}]")
            docs = vector_store.similarity_search(query, k=3, filter={"type": keyword})
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            return func(query, combined_text)

    # 일반 질문 응답 fallback
    print("[필터 없음 → 전체 문서 검색]")
    docs = vector_store.similarity_search(query, k=5)
    combined_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""당신은 뛰어난 업무 보조입니다.
다음의 [질문사항]에 대해 [참고자료]를 바탕으로 정확하고 간결하게 답변하세요.

[질문사항] 
{query}

[참고자료]
{combined_text}
"""
    llm = ChatOllama(model="qwen2.5vl:7b")
    answer = llm.invoke(prompt)
    return answer

if __name__ == "__main__":
    file_path = "sample_inputs/sample.docx"
    print(question_answer_based_vectorstore(file_path,query="퀴즈를 4개만 만들어주세요"))
