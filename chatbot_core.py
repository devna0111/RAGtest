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

def chatbot(file_path: str = "sample_inputs/sample.txt", query : str ="업로드된 파일의 내용을 한국어로 요약해주세요."):
    # 기본 파일 내용 요약 반환
    full_query = f"""당신은 한국의 신입사원들이 잘 적응하게 돕는 ai, FlowMate 입니다.
                    아래 질문에 한국어로 간단하고 문맥이 유연한 답변으로 도움을 주세요.
                    User: {query}"""
    
    answer = question_answer_based_vectorstore(file_path=file_path, query=full_query)
    return answer

if __name__ == "__main__":
    chatbot()
