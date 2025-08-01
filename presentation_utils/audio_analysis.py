import librosa
import numpy as np

def analyze_audio_features(audio_path: str) -> dict:
    """오디오 파일을 분석하여 피치, 억양, 속도 등의 음성 특성 정보를 반환하는 함수"""
    y, sr = librosa.load(audio_path) # # 오디오 파일 로드 (신호값 y, 샘플링 비율 sr)
    duration = librosa.get_duration(y=y, sr=sr) # 전체 재생 시간 계산 (초 단위)

    # 피치 분석
    # 피치 추출: piptrack은 시간-주파수 평면에서 주파수 별 에너지 기반 피치 후보 계산
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # 에너지가 중간값 이상인 피치만 추출하여 노이즈 제거
    pitch = pitches[magnitudes > np.median(magnitudes)]
    # 평균 피치 계산 (Hz 단위, 발성 높낮이 판단 기준)
    avg_pitch = np.mean(pitch) if len(pitch) > 0 else 0

    # 억양 분석 (에너지 변화) : RMS 에너지(프레임별 발화 세기)를 기반으로 변화량 계산
    frame_energy = librosa.feature.rms(y=y)[0]
    energy_var = np.var(frame_energy) # 에너지 변화 정도 (억양 변화 판단)

    # 말 속도 추정 : 박자 기반 템포 추정 (비트/분 단위, BPM 유사)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return {
        "duration_sec":duration, # 전체 길이 
        "avg_pitch": avg_pitch, # 평균 피치
        "energy_variation": energy_var, # 억양 변화량
        "speech_tempo": tempo, # 말 속도
    }


# 테스트 실행용 메인 블록
if __name__ == "__main__":
    import sys
    import os

    # 기본 테스트 경로
    default_audio = "sampleaudio.mp3"

    # 커맨드라인에서 파일 경로 받기 or 기본값
    audio_path = sys.argv[1] if len(sys.argv) > 1 else default_audio

    if not os.path.exists(audio_path):
        print(f"오디오 파일이 존재하지 않습니다: {audio_path}")
    else:
        print(f"분석 중인 오디오 파일: {audio_path}\n")
        results = analyze_audio_features(audio_path)

        print("오디오 분석 결과:")
        for k, v in results.items():
            print(f"- {k}: {v}")
