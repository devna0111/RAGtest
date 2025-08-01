# run_feedback_pipeline.py

import os
from video_processor import extract_audio, transcribe_audio, summarize_transcript
from audio_analysis import analyze_audio_features
from pose_analysis import analyze_visual_features
from feedback_generator import generate_feedback

def run_feedback_pipeline(video_path: str):
    if not os.path.exists(video_path):
        print(f"❌ 영상 파일이 존재하지 않습니다: {video_path}")
        return

    print(f"📽️ 영상 분석을 시작합니다: {video_path}")

    # 1. 오디오 추출
    audio_path = extract_audio(video_path)
    print(f"🔊 오디오 추출 완료: {audio_path}")

    # 2. STT → 전체 텍스트 변환
    transcript = transcribe_audio(audio_path)
    print("📝 변환된 발표 원고 일부:\n", transcript[:300], "...\n")

    # 3. 텍스트 요약
    summary = summarize_transcript(transcript)
    print("📌 요약 결과:\n", summary, "\n")

    # 4. 오디오 특성 분석
    audio_features = analyze_audio_features(audio_path)
    print("🎧 음성 분석 결과:")
    for k, v in audio_features.items():
        print(f"- {k}: {v}")

    # 5. 영상 기반 시각 피드백 분석
    visual_features = analyze_visual_features(video_path)
    print("🧍 시각 분석 결과:")
    for k, v in visual_features.items():
        print(f"- {k}: {v}")

    # 6. 종합 피드백 생성
    feedback = generate_feedback(summary, audio_features, visual_features)
    print("\n✅ 종합 피드백 결과:\n")
    print(feedback)


if __name__ == "__main__":
    import sys

    # 실행 인자 또는 기본 경로
    default_video = "sample_inputs/sample.mp4"
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video

    run_feedback_pipeline(video_path)
