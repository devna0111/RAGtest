from extract_docx import extract_docx_chunks_with_images
from extract_pdf import extract_pdf_chunks_with_images
from extract_img import extract_image_chunks
from extract_pptx import extract_pptx_chunks_with_images
from extract_xlsx import extract_xlsx_chunks
import os

def extract_text_and_chunks(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == '.docx':
        return extract_docx_chunks_with_images(filepath)
    elif ext == '.pdf':
        return extract_pdf_chunks_with_images(filepath)
    elif ext in ('.png', '.jpg', '.jpeg'):
        return extract_image_chunks(filepath)  # 이미지도 dict 구조로 반환됨
    elif ext in ('.ppt', '.pptx'):
        return extract_pptx_chunks_with_images(filepath)
    elif ext in ('.xlsx',):
        return extract_xlsx_chunks(filepath)
    else:
        raise ValueError(f"지원하지 않는 파일 타입: {ext}")

if __name__ == "__main__":
    # 예시: docx/pdf/pptx 모두 dict 구조로 반환됨
    # print(extract_text_and_chunks("sample.docx"))
    # print(extract_text_and_chunks("sample.pdf"))
    # print(extract_text_and_chunks("sample.pptx"))
    # print(extract_text_and_chunks("sample.xlsx"))
    print(extract_text_and_chunks("sample.png"))
