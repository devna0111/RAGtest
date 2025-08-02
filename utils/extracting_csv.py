import os
import pandas as pd

def extract_csv_content(csv_path: str, encoding: str = "utf-8") -> str:
    """
    CSV 파일에서 데이터를 읽고 마크다운 형식의 텍스트로 변환
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {csv_path}")

    # pandas로 CSV 로드 (쉼표 구분 기준)
    df = pd.read_csv(csv_path, encoding=encoding)

    # 마크다운 표 생성
    output = []
    output.append("# CSV 파일")
    output.append("[표 형식 데이터]")

    # 헤더
    headers = list(df.columns)
    output.append(" | ".join(headers))
    output.append(" | ".join(["---"] * len(headers)))

    # 각 행
    for _, row in df.iterrows():
        output.append(" | ".join(str(item) if pd.notna(item) else "" for item in row))

    return "\n".join(output)


if __name__ == "__main__":
    FILE_PATH = "sample_inputs/sample.csv"
    result = extract_csv_content(FILE_PATH)
    print("전체 추출 결과:\n", result)
