# llm_utils/report_generator.py
from langchain_ollama import ChatOllama

def generate_report_from_text(text: str, query = None) -> str:
    """
    문서 내용을 기반으로 보고서를 생성합니다. 고정 양식을 사용하고, 구체적인 보고 형식으로 구성합니다.
    """
    prompt = f"""
당신은 전문 보고서 작성 도우미입니다.
아래 문서를 참고하여 다음 조건을 만족하는 보고서를 작성해주세요:
- 제목: 문서 전체를 꿰뚫는 주제
- 항목: 개요 / 주요 내용 / 분석 / 결론 및 제언 순서
- 구어체 대신 보고서체 사용 (간결하고 단정하게)

[문서 내용]
{text}
"""
    llm = ChatOllama(model="qwen2.5vl:7b")
    return llm.invoke(prompt).content
