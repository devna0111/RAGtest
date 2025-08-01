from langchain.text_splitter import RecursiveCharacterTextSplitter
def get_adaptive_splitter(text: str) -> RecursiveCharacterTextSplitter:
    """
    전체 문서 길이에 따라 적절한 chunk_size와 overlap을 동적으로 설정
    """
    length = len(text)
    # 청크 크기 결정 (기준: 문자 수 기준)
    if length < 2000:
        chunk_size = 1000
    elif length < 5000:
        chunk_size = 1200
    elif length < 10000:
        chunk_size = 1500
    else:
        chunk_size = 2000  # 너무 크면 분산처리에 불리하므로 제한
    # overlap은 chunk_size의 10~20% 정도로 설정
    chunk_overlap = int(chunk_size * 0.15)
    print(f"[청크 설정] 전체 길이: {length}자 → chunk_size={chunk_size}, overlap={chunk_overlap}")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?"]
    )