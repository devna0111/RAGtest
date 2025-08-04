# llm_utils/summary_generator.py
from langchain_ollama import ChatOllama
def generate_summary_from_text(text: str, query = None) -> str:
        """
        문서 내용을 기반으로 요약문을 생성합니다.
        """
        prompt = f"""당신은 뛰어난 업무 보조입니다.
                다음의 [참고자료]를 바탕으로 정확하고 간결한 요약문을 만들어주세요.

                [참고자료]
                {text}
                """
        llm = ChatOllama(model="qwen2.5vl:7b")
        return llm.invoke(prompt).content