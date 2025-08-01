from dotenv import load_dotenv
load_dotenv()

FILE_PATH = 'test.pdf'

def show_metadata(docs):
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print('\n[examples]')
        max_key_lenth = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f'{k:{max_key_lenth}} : {v}')

# PDF 문서 배열 로드, 문서마다 page 번호 + page 내용
# pip install -qU pypdf
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader(FILE_PATH)
docs = loader.load()
print(docs[10].page_content[:300])

# 메타데이터
show_metadata(docs)

# OCR
# pip install -qU rapidocr-onnxruntime
# PyMuPDF : 속도 최적화, 페이지에 대한 자세한 메타데이터 포함, 페이지 -> 하나의 문서 반환
# pip install -qU pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader(FILE_PATH)
docs = loader.load()
print(docs[10].page_content[:300])
show_metadata(docs)

# Unstructured : MD, PDF 비구조화 혹은 반구조화 파일 다루는데에 특화된 인터페이스
# !pip install -qU unstructured
# Unstructured mode="elements" 지정하면 청크 단위로 반환
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader(FILE_PATH, mode="elements")
docs = loader.load()
print(docs[0].page_content)

# 데이터 카테고리 추출
set(doc.metadata["category"] for doc in docs)
show_metadata(docs)

# PyPDFium2
from langchain_community.document_loaders import PyPDFium2Loader
loader = PyPDFium2Loader(FILE_PATH)
docs = loader.load()
print(docs[10].page_content[:300])
show_metadata(docs)

# PDFMiner : HTML출력 -> BS로 파싱 -> 글꼴 크기, 페이지번호, PDF 헤더/푸터 등 
# 구조화, 풍부한 텍스트 섹션 분할
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
loader = PDFMinerPDFasHTMLLoader(FILE_PATH)
docs = loader.load()
print(docs[10].page_content[:300])
show_metadata(docs)


# PDFMiner : 문서 배열 로드, 문서마다 page 번호 + page 내용
from langchain_community.document_loaders import PDFMinerLoader
loader = PDFMinerLoader(FILE_PATH)
docs = loader.load()
print(docs[0].page_content[:300])
show_metadata(docs)

from bs4 import BeautifulSoup
soup = BeautifulSoup(docs[0].page_content, 'html.parser')
content = soup.find_all('div')

import re
cur_fs = None
cur_text = ""
snippets = []  # 동일한 글꼴 크기의 모든 스니펫 수집
for c in content:
    sp = c.find("span")
    if not sp:
        continue
    st = sp.get("style")
    if not st:
        continue
    fs = re.findall("font-size:(\d+)px", st)
    if not fs:
        continue
    fs = int(fs[0])
    if not cur_fs:
        cur_fs = fs
    if fs == cur_fs:
        cur_text += c.text
    else:
        snippets.append((cur_text, cur_fs))
        cur_fs = fs
        cur_text = c.text
snippets.append((cur_text, cur_fs))
# 중복 스니펫 제거 전략 추가 가능성 
# (PDF의 헤더/푸터가 여러 페이지에 걸쳐 나타나므로 중복 발견 시 중복 정보로 간주 가능)

from langchain_core.documents import Document

cur_idx = -1
semantic_snippets = []
# 제목 가정: 높은 글꼴 크기
for s in snippets:
    # 새 제목 판별: 현재 스니펫 글꼴 > 이전 제목 글꼴
    if (
        not semantic_snippets
        or s[1] > semantic_snippets[cur_idx].metadata["heading_font"]
    ):
        metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
        metadata.update(docs[0].metadata)
        semantic_snippets.append(Document(page_content="", metadata=metadata))
        cur_idx += 1
        continue

    # 동일 섹션 내용 판별: 현재 스니펫 글꼴 <= 이전 내용 글꼴
    if (
        not semantic_snippets[cur_idx].metadata["content_font"]
        or s[1] <= semantic_snippets[cur_idx].metadata["content_font"]
    ):
        semantic_snippets[cur_idx].page_content += s[0]
        semantic_snippets[cur_idx].metadata["content_font"] = max(
            s[1], semantic_snippets[cur_idx].metadata["content_font"]
        )
        continue

    # 새 섹션 생성 조건: 현재 스니펫 글꼴 > 이전 내용 글꼴, 이전 제목 글꼴 미만
    metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
    metadata.update(docs[0].metadata)
    semantic_snippets.append(Document(page_content="", metadata=metadata))
    cur_idx += 1

print(semantic_snippets[4])

# PDF 디렉토리로 로드
from langchain_community.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader("data/")
docs = loader.load()
print(len(docs))
print(docs[50].page_content[:300])
print(docs[50].metadata)

# PDFPlumber : PyMuPDF와 같은 기능
from langchain_community.document_loaders import PDFPlumberLoader
loader = PDFPlumberLoader(FILE_PATH)
docs = loader.load()
print(len(docs))
print(docs[10].page_content[:300])
show_metadata(docs)



