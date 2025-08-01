# pip install langchain-community pdfplumber pymupdf python-docx

import os
import fitz  # PyMuPDF
import pdfplumber
from io import StringIO

def extract_pdf_all_in_order_as_string(pdf_path: str) -> str:
    output = StringIO()

    output.write("# 자동 생성\n\n")

    pdf_fitz = fitz.open(pdf_path)
    pdf_plumber = pdfplumber.open(pdf_path)

    for page_num in range(len(pdf_fitz)):
        output.write(f"## 페이지 {page_num + 1}\n\n")

        # 텍스트 추출
        text = pdf_plumber.pages[page_num].extract_text()
        if text:
            output.write("**본문 텍스트:**\n")
            output.write(text.strip() + "\n\n")

        # 표 추출
        tables = pdf_plumber.pages[page_num].extract_tables()
        for t_idx, table in enumerate(tables):
            output.write(f"**[표 {t_idx + 1}]**\n")
            if table:
                for row in table:
                    row_text = " | ".join(cell if cell else "" for cell in row)
                    output.write(row_text + "\n")
                output.write("\n")

        # 이미지 설명만
        page = pdf_fitz[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            output.write(f"[이미지 {img_index + 1}] 페이지 내 포함된 이미지] 실제사진이미지(그래프,텍스트아님)\n")

        output.write("\n---\n\n")

    pdf_plumber.close()
    pdf_fitz.close()

    return output.getvalue()


if __name__ == "__main__":
    FILE_PATH = "test.pdf"  # 변환할 PDF 파일명
    # OUTPUT_PATH = "output_report.md"  # 저장할 마크다운 파일명

    result = extract_pdf_all_in_order_as_string(FILE_PATH)

    # 출력
    print(result)

    # # 저장
    # with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    #     f.write(result)

    # print(f"[저장 완료] {OUTPUT_PATH}")
