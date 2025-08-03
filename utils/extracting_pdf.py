import os
import fitz  # PyMuPDF
import pdfplumber
from io import StringIO
from utils.image_utils import analyze_image_with_qwen

def extract_pdf_all_in_order_as_string(pdf_path: str, mode: str = "simple") -> str:
    output = StringIO()
    image_output_dir = "temp_imgs"
    os.makedirs(image_output_dir, exist_ok=True)

    output.write("# PDF 자동 변환\n\n")

    pdf_fitz = fitz.open(pdf_path)
    pdf_plumber = pdfplumber.open(pdf_path)

    img_idx = 1

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

        # 이미지 추출 및 분석
        page = pdf_fitz[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_fitz.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_path = os.path.join(image_output_dir, f"page_{page_num+1}_img_{img_idx}.{ext}")

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            try:
                result = analyze_image_with_qwen(image_path, mode=mode)
                output.write(f"**[이미지 {img_idx} 분석 결과]**\n{result.strip()}\n\n")
            except Exception as e:
                output.write(f"[이미지 분석 실패: {e}]\n\n")

            img_idx += 1

        output.write("---\n\n")

    pdf_plumber.close()
    pdf_fitz.close()

    return output.getvalue()


if __name__ == "__main__":
    FILE_PATH = "sample_inputs/sample.pdf"  # 변환할 PDF 파일 경로
    result = extract_pdf_all_in_order_as_string(FILE_PATH, mode="simple")
    print(result)
