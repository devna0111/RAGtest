from PIL import Image
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="korean")

def extract_image_chunks(filepath):
    result = ocr.predict(filepath)
    lines = result[0]['rec_texts'] if result else []
    text = "\n".join(lines)
    return [{"type": "ocr", "content": text}] if text else []

if __name__ == "__main__" :
    print(extract_image_chunks("sample4.png"))
    # print(ocr.predict("imgsample.png")[0]['rec_texts'])