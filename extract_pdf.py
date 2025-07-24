import pdfplumber
import os
from PIL import Image
import io
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="korean")

def extract_pdf_chunks_with_images(filepath, img_dir='img_output_pdf'):
    os.makedirs(img_dir, exist_ok=True)
    chunks = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            # 텍스트
            text = page.extract_text()
            if text:
                chunks.append({"type": "text", "content": text})

            # 표(기본: 텍스트화) - pdfplumber tables
            try:
                tables = page.extract_tables()
                for t in tables:
                    table_lines = ["\t".join([cell if cell else "" for cell in row]) for row in t]
                    tstr = "\n".join(table_lines)
                    if tstr.strip():
                        chunks.append({"type": "table", "content": tstr})
            except Exception:
                pass

            # 이미지 추출 + OCR
            try:
                for img_idx, img_dict in enumerate(page.images):
                    # pdfplumber images: {"x0","y0","x1","y1","width","height","stream"}
                    if "stream" in img_dict:
                        img_stream = img_dict["stream"].get_data()
                        img = Image.open(io.BytesIO(img_stream))
                        img_path = os.path.join(img_dir, f"page{i}_img{img_idx}.png")
                        img.save(img_path)
                        chunks.append({"type": "image", "content": img_path})

                        result = ocr.predict(img_path)
                        ocr_lines = result[0]['rec_texts'] if result else []
                        ocr_text = "\n".join(ocr_lines)
                        if ocr_text:
                            chunks.append({"type": "ocr", "content": ocr_text})
            except Exception:
                pass

    return chunks

if __name__ == "__main__":
    out = extract_pdf_chunks_with_images("샘플.pdf")
    for chunk in out:
        print(f"[{chunk['type']}]", chunk['content'][:80])
