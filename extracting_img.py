from PIL import Image
import base64
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import os

def image_to_base64(image_path: str) -> str:
    """이미지를 base64 문자열로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image_with_qwen(image_path: str, prompt: str) -> str:
    """Langchain + Ollama 기반 Qwen2.5-VL 분석"""
    base64_img = image_to_base64(image_path)

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

    prompt = (
    "이 이미지를 보고 다음 항목을 분석해줘. 특히 숫자나 수치가 포함된 내용은 절대 생략하지 말고 가능한 한 **정확하게 모두** 추출해줘.\n\n"
    "1. [텍스트 전체 추출] 코드나 출력 내용 등 모든 텍스트를 원문 그대로 추출해줘. 수치는 생략 없이 빠짐없이 포함해야 해.\n"
    "2. [표 구조 분석] 표가 있다면, 표의 행과 열을 설명하고, 각 셀의 데이터를 가능한 한 정확하게 복원해줘.\n"
    "3. [그래프 분석] 그래프나 차트가 있다면 축의 이름, 숫자 범위, 추세 등을 설명해줘.\n"
    "4. [이미지 또는 도식 구조] 다이어그램이나 흐름도가 있다면 구조적 관계를 설명해줘.\n"
    "5. [추론 및 해석] 위 정보를 기반으로 유추 가능한 의미, 데이터 흐름, 의도를 설명해줘.\n\n"
    "❗ 숫자, 수치, 배열, 행렬 등은 생략 없이 모두 보여줘. 특히 `shape` 출력처럼 괄호로 된 숫자 묶음이 여러 개 있는 경우, 하나도 빠짐없이 나열해줘."
)


    result = analyze_image_with_qwen(image_path, prompt)
    print("✅ 이미지 요약 결과:\n", result)
