from utils.chunk_utils import get_adaptive_splitter
from utils.extracting_docx import extract_docx_content
from utils.extracting_img import analyze_image_with_qwen
from utils.extracting_pdf import extract_pdf_all_in_order_as_string
from utils.extracting_pptx import pptx_to_markdown_string
from langchain_core.documents import Document
from utils.extracting_xlsx import extract_xlsx_content
from utils.extracting_csv import extract_csv_content
from utils.extracting_txt import extract_txt_content
from presentation_utils.video_processor import transcribe_audio, extract_audio
from utils.wholetext_summary import make_summary
from utils.wholetext_toPresentation import make_presentation

def start_extracting(file_path: str) -> str:
    """
    확장자에 따라 알맞은 추출 함수 호출
    추출 결과는 청크 이전 단계인 str 전체 텍스트
    """
    ext = file_path.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg', 'png']:
        return analyze_image_with_qwen(file_path)
    elif ext in ['doc','docx']:
        return extract_docx_content(file_path)
    elif ext == 'pdf':
        return extract_pdf_all_in_order_as_string(file_path)
    elif ext in ['ppt','pptx']:
        return pptx_to_markdown_string(file_path)
    elif ext in ['xlsx', 'xls']:
        return extract_xlsx_content(file_path)
    elif ext == 'txt':
        return extract_txt_content(file_path)
    elif ext == 'csv' :
        return extract_csv_content(file_path)
    elif ext in ["wav","mp3"] :
        return transcribe_audio(file_path)
    elif ext in ["mp4"] :
        audio = extract_audio(file_path)
        return transcribe_audio(audio)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

def split_chunks(file_path: str) -> list:
    """
    1. 요약문
    2. 발표문
    3. 문서에서 추출한 전체 텍스트를 의미 있는 청크 단위로 나눈 결과,
    LangChain Document 리스트로 반환
    """
    whole_text = start_extracting(file_path)
    summary_text = make_summary(whole_text)
    presentation_text = make_presentation(whole_text)
    text_splitter = get_adaptive_splitter(whole_text)
    chunks = text_splitter.split_text(whole_text)
    result = [Document(page_content=summary_text, metadata={"source": file_path}),
            Document(page_content=presentation_text, metadata={"source": file_path}),]
    for chunk in chunks :
        result.append(Document(page_content=chunk, metadata={"source": file_path}))
    return result



if __name__ == "__main__":
    test_file_path = "sample_inputs/sample.docx"  # 또는 sample.pdf, sample.png, sample.pptx 등

    # 청크 분리 실행
    try:
        documents = split_chunks(test_file_path)

        print(f"\n총 {len(documents)}개의 청크가 생성")
        for i, doc in enumerate(documents):
            print(f"\n--- 청크 {i + 1} ---")
            print(doc.page_content)

    except Exception as e:
        print(f"\n[에러 발생] {str(e)}")
