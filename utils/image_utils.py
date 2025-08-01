import base64
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def image_to_base64(image_path: str) -> str:
    """
    이미지 파일을 base64 문자열로 인코딩
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image_with_qwen(image_path: str, model: str = "qwen2.5vl:7b") -> str:
    """
    Qwen2.5-VL 모델을 통해 이미지 분석 결과를 반환
    """
    IMAGE_ANALYSIS_PROMPT = (
    "이 이미지를 보고 다음 항목을 분석해줘. 특히 숫자나 수치가 포함된 내용은 절대 생략하지 말고 가능한 한 정확하게 모두 추출해줘.\n\n"
    "1. [텍스트 전체 추출] 코드나 출력 내용 등 모든 텍스트를 원문 그대로 추출해줘.\n"
    "2. [표 구조 분석] 표가 있다면, 각 셀의 데이터를 가능한 한 정확하게 복원해줘.\n"
    "3. [그래프 분석] 축의 이름, 숫자 범위, 추세 등을 설명해줘.\n"
    "4. [도식 구조] 다이어그램이나 흐름도가 있다면 구조적 관계를 설명해줘.\n"
    "숫자, 수치, 배열 등은 생략 없이 모두 보여줘."
    )
    base64_img = image_to_base64(image_path)
    llm = ChatOllama(model="qwen2.5vl:7b")
    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
            {"type": "text", "text": IMAGE_ANALYSIS_PROMPT}
        ]
    )
    response = llm.invoke([message])
    return response.content
