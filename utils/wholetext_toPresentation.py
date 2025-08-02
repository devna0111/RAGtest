from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

llm = ChatOllama(model='qwen2.5vl:7b')

def make_presentation(wholetext: str) -> str:
    length = len(wholetext)

    # 메시지 생성 함수 (Runnable로 wrapping)
    def generate_messages(_: str):
        return [
            SystemMessage(
                content=(
                    "당신은 문서를 입력 받으면 핵심을 중심으로 발표안을 만드는 전문가입니다.\n"
                    f"입력받은 문서의 길이는 {length} 자입니다.\n"
                    f"전문성이 보이는 발표문을 만들어주세요. 약 10분에서 15분 정도 발표합니다."
                    "자연스럽게 문맥이 이어지는 구어체로 만들어주세요."
                )
            ),
            HumanMessage(
                content=(
                    "다음 문서 [text]를 발표문으로 만들어주세요."
                    "주어진 텍스트만을 이용합니다.\n\n"
                    f"[text]\n{wholetext}"
                )
            )
        ]

    # Runnable 체인 구성
    summary_chain = RunnableLambda(generate_messages) | llm

    # 실행
    response = summary_chain.invoke("")

    return response.content

if __name__ == "__main__":
    sample_text = "이 문서는 사내 업무 자동화를 위한 가이드입니다. 주요 내용은 RAG기반 문서 처리와 요약, 벡터 DB 저장 등을 포함합니다..."
    presentation = make_presentation(sample_text)
    print("결과:\n", presentation)
