from transformers import DonutProcessor, VisionEncoderDecoderModel, pipeline
from PIL import Image

# 1. Processor, Model 직접 준비 (use_fast=True 명시)
processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-korean",
    use_fast=True
)
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-korean")

# pipeline에서 tokenizer=processor.tokenizer로 전달!
pipe = pipeline(
    "document-question-answering",
    model=model,
    tokenizer=processor.tokenizer,             # ★ 이걸로!
    image_processor=processor.image_processor,
    # past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values),
    device=0 # 0 : gpu / -1 : cpu
)
# cmd : set PYTORCH_SDP_ATTENTION_BACKEND=eager
def extract_structured_doc(image_path):
    image = Image.open(image_path)
    prompt = (
    "<s_docvqa><s_question>"
    "이미지에 포함된 **모든 텍스트, 표, 그래프, 사진, 도형, 그림**의 내용을 빠짐없이, "
    "원본의 구조와 정보 손실 없이 최대한 자세히, 구분해서 추출해서 반환해줘."
    "원본의 순서와 레이아웃을 최대한 반영해"
    "누락 없이, 요약 없이, 전체 정보를 반환해"
    "표/그래프/사진은 따로 구분해서 마크다운 표/설명/캡션 형식으로 반환해"
    "<s_answer>"
)
    result = pipe(image=image, question=prompt)
    return result

if __name__ == "__main__":
    print(extract_structured_doc("imgsample.png"))
