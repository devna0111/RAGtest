# utils/quiz_generator.py
from langchain_ollama import ChatOllama
import re

def extract_num_questions(query: str) -> int:
    """
    자연어 쿼리에서 문제 수를 추출합니다. 없으면 기본값 반환.
    예: "퀴즈 5문제 만들어줘" → 5
    """
    match = re.search(r"(퀴즈|문제)[^\d]{0,5}(\d+)", query)
    return int(match.group(2)) if match else 5

def generate_quiz_from_text(text: str, query: str = "퀴즈를 만들어줘") -> str:
    """
    문서 내용을 기반으로 이해도 체크용 퀴즈를 생성합니다.
    - query 내에 문제 수가 포함되어 있으면 반영합니다.
    - 각 문항은 보기 4개 + 정답 포함
    """
    num_questions = extract_num_questions(query)

    prompt = f"""
당신은 교육 자료를 기반으로 퀴즈를 만드는 전문가입니다.

아래 문서를 바탕으로 이해도 확인을 위한 객관식 문제 {num_questions}개를 만들어주세요.
- 각 문항은 보기 4개를 포함하고
- 반드시 정답을 함께 명시해주세요.

[문서 내용]
{text}
"""
    llm = ChatOllama(model="qwen2.5vl:7b")
    return llm.invoke(prompt).content


if __name__ == "__main__":
    test_text = "이 문서는 사내 업무 자동화를 위한 가이드입니다. 주요 내용은 RAG기반 문서 처리와 요약, 벡터 DB 저장 등을 포함합니다."
    test_query = "5문제 퀴즈 만들어줘"
    print("[🧠 퀴즈 생성 결과]\n")
    print(generate_quiz_from_text(test_text, test_query))