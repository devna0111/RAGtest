import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

class RAGPipeline:
    """
    RAG 파이프라인의 모든 구성요소를 관리하는 클래스.
    - 벡터 DB 초기화 및 로드
    - 텍스트 추가 및 임베딩
    - 질의응답 체인 실행
    """
    def __init__(self, db_path, text_model, embed_model):
        self.db_path = db_path
        self.text_model_name = text_model
        self.embed_model_name = embed_model
        
        # 모델 초기화
        self.llm = Ollama(model=self.text_model_name)
        self.embeddings = OllamaEmbeddings(model=self.embed_model_name)
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # 벡터 DB 초기화
        self.vectordb = self._load_or_create_vectordb()

    def _load_or_create_vectordb(self):
        """디스크에 DB가 있으면 로드하고, 없으면 새로 생성합니다."""
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            print(f"기존 벡터 DB를 로드합니다: {self.db_path}")
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        else:
            print("새로운 벡터 DB를 생성합니다.")
            # 임시 문서를 넣어 DB를 생성하고 바로 persist
            dummy_doc = self.text_splitter.create_documents(["init"])
            db = Chroma.from_documents(documents=dummy_doc, embedding=self.embeddings, persist_directory=self.db_path)
            db.persist()
            # 생성 후 다시 로드하여 일관성 유지
            return Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

    def add_texts_to_db(self, texts: list[str]):
        """추출된 텍스트 목록을 분할하고 벡터 DB에 추가합니다."""
        if not texts:
            print("DB에 추가할 텍스트가 없습니다.")
            return
        
        print(f"{len(texts)}개의 텍스트 조각을 DB에 추가합니다.")
        documents = self.text_splitter.create_documents(texts)
        self.vectordb.add_documents(documents)
        self.vectordb.persist()
        print("DB 저장이 완료되었습니다.")

    def query(self, question: str) -> dict:
        """사용자 질문에 대한 답변을 생성합니다."""
        if self.vectordb is None:
            return {"result": "벡터 DB가 초기화되지 않았습니다. 문서를 먼저 업로드해주세요."}

        retriever = self.vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": question})
        return result