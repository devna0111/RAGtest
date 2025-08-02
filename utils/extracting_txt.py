import os

def extract_txt_content(txt_path: str) -> str:
    """
    텍스트(.txt) 파일에서 전체 줄글을 추출하고 마크다운 포맷으로 정리하여 반환
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 마크다운 형식으로 감싸기
    output = []
    output.append("# 텍스트 파일")
    output.append("[본문 내용]")
    output.append(text.strip())

    return "\n\n".join(output)


if __name__ == "__main__":
    FILE_PATH = "sample_inputs/sample.txt"
    result = extract_txt_content(FILE_PATH)
    print("전체 추출 결과:\n", result)