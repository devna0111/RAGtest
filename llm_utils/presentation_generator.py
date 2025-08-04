# llm_utils/presentation_generator.py
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

def generate_presentation_from_text(text: str,query = None) -> str:
    """
    입력된 텍스트를 바탕으로 발표문 형태로 구성합니다.
    - 발표 시간은 약 10~15분 분량 기준
    - 발표체 / 구어체 스타일
    """
    length = len(text)

    def build_messages(_: str):
        return [
            SystemMessage(
                content=(
                    "당신은 문서를 입력받아 핵심을 중심으로 발표문을 만드는 전문가입니다.\n"
                    f"입력된 문서의 길이는 약 {length}자입니다.\n"
                    "약 10~15분 분량의 발표문을 구어체로 자연스럽게 작성해주세요."
                )
            ),
            HumanMessage(
                content=(
                    "다음 문서를 발표문으로 작성해주세요.\n"
                    "[문서 내용]\n"
                    f"{text}"
                )
            )
        ]

    llm = ChatOllama(model="qwen2.5vl:7b")
    chain = RunnableLambda(build_messages) | llm
    response = chain.invoke("")
    return response.content


if __name__ == "__main__":
    sample_text = "이 문서는 사내 업무 자동화를 위한 가이드입니다. 주요 내용은 RAG기반 문서 처리와 요약, 벡터 DB 저장 등을 포함합니다..."
    result = generate_presentation_from_text(sample_text)
    print("\n[발표문 결과]\n", result)
