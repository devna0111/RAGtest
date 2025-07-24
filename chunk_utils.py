# overlap, chunking utility

def add_overlap(chunks, overlap=100) :
    ''' 추출한 텍스트 chunks 사이에 overlap 추가'''
    if not chunks or overlap<= 0 :
        return chunks
    overlapped_chunks = []
    prev_chunk = ""
    for chunk in chunks :
        if prev_chunk :
            ov = prev_chunk[-overlap:]
            merged = ov + chunk
            overlapped_chunks.append(merged)
        else :
            overlapped_chunks.append(chunk)
    return overlapped_chunks

if __name__ == "__main__":
    ex = ["이것은 첫 번째 청크입니다.", "두 번째 청크이고 이전 내용이 겹칩니다.", "세 번째 청크."]
    print(add_overlap(ex, overlap=10))