from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

llm = ChatOllama(model='qwen2.5vl:7b')

def make_summary(wholetext: str) -> str:
    length = len(wholetext)

    if length < 2000:
        chunk_size = 1000
    elif length < 5000:
        chunk_size = 1200
    elif length < 10000:
        chunk_size = 1500
    else:
        chunk_size = 2000

    # 메시지 생성 함수 (Runnable로 wrapping)
    def generate_messages(_: str):
        return [
            SystemMessage(
                content=(
                    "당신은 문서를 받으면 핵심을 요약하는 데 특화된 전문가입니다.\n"
                    f"입력받은 문서의 길이는 {length} 자입니다.\n"
                    f"문서 요약은 1개 청크로 이루어져야 하며 {chunk_size}자 이내로 요약해주세요."
                )
            ),
            HumanMessage(
                content=(
                    "다음 문서 [text]를 요약해주세요. 이모지 등의 표현은 일체 사용하지 않으며 "
                    "중복되는 내용은 없어야 합니다."
                    "주어진 텍스트만을 이용합니다.\n\n"
                    f"[text]\n{wholetext}"
                )
            )
        ]

    # Runnable 체인 구성
    summary_chain = RunnableLambda(generate_messages) | llm

    # 실행
    response = summary_chain.invoke("")

    return "[요약문]" + response.content

if __name__ == "__main__":
    sample_text = "이 문서는 사내 업무 자동화를 위한 가이드입니다. 주요 내용은 RAG기반 문서 처리와 요약, 벡터 DB 저장 등을 포함합니다..."
    summary = make_summary(sample_text)
    print("요약 결과:\n", summary)
