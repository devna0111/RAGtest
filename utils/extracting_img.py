from PIL import Image
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import os

def image_to_base64(image_path: str) -> str:
    """이미지를 base64 문자열로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image_with_qwen(image_path: str) -> str:
    """Langchain + Ollama 기반 Qwen2.5-VL 분석"""
    base64_img = image_to_base64(image_path)

    prompt = (
        "이 이미지를 보고 다음 항목을 분석해줘. 특히 숫자나 수치가 포함된 내용은 절대 생략하지 말고 가능한 한 정확하게 모두 추출해줘.\n\n"
    "1. [텍스트 전체 추출] 코드나 출력 내용 등 모든 텍스트를 원문 그대로 추출해줘.\n"
    "2. [표 구조 분석] 표가 있다면, 각 셀의 데이터를 가능한 한 정확하게 복원해줘.\n"
    "3. [그래프 분석] 축의 이름, 숫자 범위, 추세 등을 설명해줘.\n"
    "4. [도식 구조] 다이어그램이나 흐름도가 있다면 구조적 관계를 설명해줘.\n"
    "숫자, 수치, 배열 등은 생략 없이 모두 보여줘.",
    "만약 1, 2, 3, 4번 중 해당하는 내용이 없는 항목이 있다면 그 부분은 제외하고 확인한 내용만 가감 없이 정보만 전달해줘"
    )
    
    # Ollama multimodal 모델 호출
    llm = ChatOllama(
        model="qwen2.5vl:7b",
    )

    # Langchain 멀티모달 메시지 구성
    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
            {"type": "text", "text": prompt}
        ]
    )

    response = llm.invoke([message])
    return response.content


if __name__ == "__main__":
    image_path = "sample.png"


    result = analyze_image_with_qwen(image_path)
    print("✅ 이미지 요약 결과:\n", result)
