import os
import hashlib
from parsing_utils import split_chunks
from langchain_qdrant import Qdrant
from langchain_ollama import OllamaEmbeddings, ChatOllama
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
# from llm_utils.quiz_generator import generate_quiz_from_text
# from llm_utils.report_generator import generate_report_from_text
# from llm_utils.summary_generator import generate_summary_from_text
# from llm_utils.presentation_generator import generate_presentation_from_text

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
        print(f"[이미 존재하는 컬렉션: {collection_name}] → 기존 DB 사용")
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

def question_answer_based_vectorstore(file_path='sample_inputs/sample.txt', query: str = "문서를 요약해주세요") -> str:
    vector_store = data_to_vectorstore(file_path)

    # 특정 필터를 사용하면 query 검색량을 폭발적으로 증가시켜야함.
    # 퀴즈, 보고서, 발표, 요약을 요청하면 검색량을 늘림
    # 그런데 이때 이전 내용을 히스토리로 갖고 이걸 토대로 대화하면 검색량이 다시 늘어나는 문제점이 발생함.
    # 이건 어떻게 해결하지?
    filter = ['요약','발표','퀴즈','보고서']
    for func in filter :
        if func in query:
            docs = vector_store.similarity_search(query,k=1000,)
            # print(docs[0].page_content)
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""당신은 뛰어난 업무 보조입니다.
                        다음의 [질문사항]에 대해 [참고자료]를 바탕으로 정확하고 간결하게 답변하세요.

                        [질문사항] 
                        {query}

                        [참고자료]
                        {combined_text}
                        """
            llm = ChatOllama(model='qwen2.5vl:7b', repeat_penalty=1.15, temperature=0.2)
            answer = llm.invoke(prompt)
            return answer.content

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
    llm = ChatOllama(model='qwen2.5vl:7b', repeat_penalty=1.15, temperature=0.2)
    answer = llm.invoke(prompt)
    return answer.content

if __name__ == "__main__":
    file_path = "sample_inputs/sample.docx"
    while True :
        query = input('명령어를 입력하세요 (종료:끝) : ')
        if query in ['끝','종료'] :
            break
        test = question_answer_based_vectorstore(file_path,query=query)
        print(test)
        if '보고서' in query :
            from llm_utils import docx_writer
            docx_writer.markdown_to_styled_docx(test)
            print("보고서 초안이 작성완료되었습니다.")
        elif '발표자료' in query :
            from llm_utils import pptx_writer
            pptx_writer.save_structured_text_to_pptx(test)
