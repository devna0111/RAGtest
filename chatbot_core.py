from vectordb_upload_search import question_answer_based_vectorstore

class BufferMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []

    def append(self, user, assistant):
        self.history.append({"user": user, "assistant": assistant})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_formatted_history(self):
        # LLM에 넣을 때 사용
        return "\n".join(
            [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in self.history]
        )

def chatbot(file_path: str = "sample_inputs/sample.txt"):
    print("챗봇을 시작합니다. 종료를 원하시면 'exit', 'bye', '끝', '종료' 중 하나를 입력하세요.")
    memory = BufferMemory(max_turns=5)  # 최근 5턴만 기억
    while True:
        query = input("🤖 어떤 내용이 궁금하세요? : ").strip()
        if query.lower() in ["exit", "bye", "끝", "종료"]:
            print("챗봇을 종료합니다.")
            break

        # 맥락을 함께 보냄
        context = memory.get_formatted_history()
        full_query = f"당신은 한국의 신입사원들이 잘 적응하게 돕는 ai, FlowMate 입니다. 아래 질문에 간단하게 답변으로 도움을 주세요! {context}\nUser: {query}" if context else query

        answer = question_answer_based_vectorstore(file_path=file_path, query=full_query)
        print("💬 답변:", answer)
        memory.append(query, answer)

    print("\n📚 대화 로그 요약:")
    for i, h in enumerate(memory.history, 1):
        print(f"{i}. Q: {h['user']} → A: {h['assistant']}")

if __name__ == "__main__":
    chatbot()
