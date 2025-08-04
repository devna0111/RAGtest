from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def extract_audio(video_path: str, output_path="temp_wav/temp_audio.wav"):
    """
    영상에서 오디오만 추출하여 WAV 파일로 저장하는 함수입니다.
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path)  # verbose=False, logger=None 제거
    return output_path

def transcribe_audio(audio_path: str) -> str:
    """
    faster-whisper 모델을 사용하여 음성 파일을 텍스트로 변환합니다.
    """
    model = WhisperModel("large-v2", device="cuda", compute_type="float16")  # 또는 "cpu", "int8"

    segments, info = model.transcribe(audio_path, language="ko")

    # segment는 generator이므로 반복문으로 텍스트 추출
    result = " ".join([seg.text for seg in segments])
    return result

def summarize_transcript(transcript: str) -> str:
    """
    Langchain + Ollama를 사용하여 텍스트(발표 원고)를 요약합니다.
    """
    prompt = ChatPromptTemplate.from_template(
        "다음 발표 원고 내용을 간결하고 핵심 위주로 요약해줘:\n\n{transcript}"
    )
    llm = ChatOllama(model="qwen2.5vl:7b")
    chain = prompt | llm
    return chain.invoke({"transcript": transcript})


if __name__ == "__main__":
    import sys
    import os

    # 기본 테스트 영상 경로
    default_video = "uploads/sample.mp4"
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video

    if not os.path.exists(video_path):
        print(f"오류: 영상 파일이 존재하지 않습니다 → {video_path}")
    else:
        print(f"영상 파일을 분석합니다: {video_path}")

        # 1. 오디오 추출
        audio_path = extract_audio(video_path)
        print("오디오 추출 완료:", audio_path)

        # 2. STT 수행
        transcript = transcribe_audio(audio_path)
        print("텍스트 변환 결과 (일부):")
        print(transcript[:300], "...")  # 전체 출력이 너무 길 경우 앞부분만 출력

        # 3. 요약
        summary = summarize_transcript(transcript)
        print("\n요약 결과:")
        print(summary)
