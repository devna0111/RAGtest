from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def generate_feedback(summary: str, audio_features: dict, visual_features: dict) -> str:
    prompt_template = ChatPromptTemplate.from_template("""
[발표 요약]
{summary}

[음성 분석 결과]
- 총 길이: {duration_sec}초
- 평균 피치: {avg_pitch}
- 억양 변화량: {energy_variation}
- 말의 속도 (추정): {speech_tempo}

[시각 표현 분석 결과]
- 얼굴 인식 비율: {face_detection_ratio}
- 제스처 사용 비율: {gesture_ratio}

위 내용을 바탕으로 발표에 대한 종합 피드백을 제공해줘.
- 발표 내용 구성과 전달력
- 발화 특성 (속도, 억양 등)
- 비언어적 표현 (표정, 제스처 등)
- 개선 포인트와 잘한 점을 구분해줘.
- 다만, 수치가 0이라면 인식하지 못했으니 다른 영상을 업로드 요청하도록 유도해줘.
""")
    llm = ChatOllama(model="qwen2.5vl:7b")
    chain = prompt_template | llm
    return chain.invoke({
        **audio_features,
        **visual_features,
        "summary": summary
    })

if __name__ == "__main__":
    dummy_summary = "본 발표는 서울시 공유 킥보드 운영 개선 방안에 대해 다루었습니다. 수요 예측, 운영 효율화, 데이터 기반 정책 방향 등을 포함했습니다."
    
    dummy_audio_features = {
        "duration_sec": 180,
        "avg_pitch": "220Hz",
        "energy_variation": "중간",
        "speech_tempo": "빠름"
    }

    dummy_visual_features = {
        "face_detection_ratio": "85%",
        "gesture_ratio": "70%"
    }

    feedback = generate_feedback(dummy_summary, dummy_audio_features, dummy_visual_features)
    print("🎤 발표 피드백 결과:\n", feedback.content)
