# chatbot으로 대화를 한 내용을 어떻게 저장할까?
# http://localhost:6333/dashboard⁠

# 1. MySQL

# 2. 벡터 DB 업로드로 대충 떼우기

# 3. 그냥 캐시메모리로 띄워놓기?
# chatbot_agent.py

from vectordb_upload_search import question_answer_based_vectorstore
from collections import deque

class BufferMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def append(self, user, assistant):
        self.history.append({"user": user, "assistant": assistant})

    def get_context(self):
        return "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history])

class FlowMateChatbot:
    def __init__(self, file_path: str = "sample_inputs/sample.txt", max_turns: int = 5):
        self.file_path = file_path
        self.memory = BufferMemory(max_turns=max_turns)
        print(f"📂 문서 로딩 및 벡터화 준비 완료: {file_path}")

    def _format_query(self, user_query: str) -> str:
        context = self.memory.get_context()
        return f"""당신은 신입사원이 사내 문서를 잘 이해하도록 돕는 AI FlowMate입니다.
아래 대화 문맥과 현재 질문을 참고하여 정확하고 간결하게 답변해주세요.

{context if context else ""}
User: {user_query}"""

    def ask(self, query: str) -> str:
        full_query = self._format_query(query)
        answer = question_answer_based_vectorstore(file_path=self.file_path, query=full_query)
        self.memory.append(query, answer)
        return answer

    def chat_loop(self):
        print("💬 FlowMate 챗봇을 시작합니다. 종료하려면 'exit', 'bye', '끝', '종료'를 입력하세요.")
        while True:
            user_input = input("👤 질문: ").strip()
            if user_input.lower() in ["exit", "bye", "끝", "종료"]:
                print("🛑 FlowMate 챗봇을 종료합니다.")
                break
            response = self.ask(user_input)
            print("🤖 답변:", response)

        print("\n📚 대화 히스토리:")
        for i, h in enumerate(self.memory.history, 1):
            print(f"{i}. Q: {h['user']} → A: {h['assistant']}")


# 실행 테스트
if __name__ == "__main__":
    bot = FlowMateChatbot()
    bot.chat_loop()