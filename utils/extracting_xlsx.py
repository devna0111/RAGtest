import os
import time
import hashlib
import openpyxl
import xlrd
from openpyxl.drawing.image import Image as XLImage
from utils.image_utils import analyze_image_with_qwen
from concurrent.futures import ThreadPoolExecutor

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_xlsx_content(file_path: str, enable_image_analysis: bool = True) -> str:
    ext = file_path.split('.')[-1].lower()
    content_list = []
    image_output_dir = "temp_imgs"
    os.makedirs(image_output_dir, exist_ok=True)

    image_cache = {}
    image_futures = {}
    img_idx = 1

    if ext == "xlsx":
        wb = openpyxl.load_workbook(file_path, data_only=True)

        for sheet in wb.worksheets:
            content_list.append(f"# 시트: {sheet.title}")
            table_text = []

            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                table_text.append(row_text)

            if table_text:
                content_list.append("[표 또는 셀 텍스트]")
                content_list.append("\n".join(table_text))

            if enable_image_analysis:
                for img in sheet._images:
                    if isinstance(img, XLImage):
                        print(f"\n[이미지 {img_idx}] 처리 시작")

                        # 이미지 저장 시간 측정
                        t1 = time.time()
                        img_data = img._data()
                        img_path = os.path.join(image_output_dir, f"image_{img_idx}.png")
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        t2 = time.time()
                        print(f"[이미지 {img_idx}] 저장 완료 (소요 시간: {t2 - t1:.2f}초)")

                        img_hash = get_image_hash(img_path)

                        if img_hash not in image_cache:
                            # 분석 시간 측정
                            t3 = time.time()
                            summary = analyze_image_with_qwen(img_path)
                            t4 = time.time()
                            print(f"[이미지 {img_idx}] Qwen 분석 완료 (소요 시간: {t4 - t3:.2f}초)")
                            image_cache[img_hash] = summary
                        else:
                            summary = image_cache[img_hash]
                            print(f"[이미지 {img_idx}] 캐시된 결과 사용")

                        content_list.append(f"[이미지{img_idx}] Qwen 분석 결과:\n{summary.strip()}")
                        img_idx += 1

    elif ext == "xls":
        wb = xlrd.open_workbook(file_path)

        for sheet in wb.sheets():
            content_list.append(f"# 시트: {sheet.name}")
            table_text = []

            for row_idx in range(sheet.nrows):
                row = sheet.row_values(row_idx)
                row_text = " | ".join(str(cell) if cell != "" else "" for cell in row)
                table_text.append(row_text)

            if table_text:
                content_list.append("[표 또는 셀 텍스트]")
                content_list.append("\n".join(table_text))

    else:
        raise ValueError("지원하지 않는 엑셀 형식입니다. xlsx 또는 xls 파일만 가능합니다.")

    return "\n\n".join(content_list)


if __name__ == "__main__":
    FILE_PATH = "sample_inputs/sample.xlsx"
    result = extract_xlsx_content(FILE_PATH, enable_image_analysis=True)
    print("\n전체 추출 결과:\n", result)
