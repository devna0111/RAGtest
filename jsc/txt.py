def extract_txt_file_as_string(txt_path: str) -> str:
    """TXT 파일에서 전체 텍스트를 읽어 문자열로 반환"""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(txt_path, "r", encoding="cp949") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"인코딩 문제로 파일을 읽을 수 없습니다: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {txt_path}")
    except Exception as e:
        raise RuntimeError(f"텍스트 파일을 읽는 중 예외 발생: {e}")


if __name__ == "__main__":
    for file in ["인코딩에러.txt"]:
        print(f"\n파일: {file}")
        try:
            result = extract_txt_file_as_string(file)
            print("내용:\n", result)
        except Exception as e:
            print("오류 발생:", e)
