import cv2
import mediapipe as mp

# 🎥 얼굴, 제스처 등 비언어적 표현을 영상에서 분석
def analyze_visual_features(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)

    # MediaPipe 모듈 초기화
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    mp_pose = mp.solutions.pose.Pose()
    mp_hands = mp.solutions.hands.Hands()

    # 통계 변수 초기화
    frame_count = 0
    face_detected = 0
    gesture_detected = 0

    # 프레임 단위로 영상 분석
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV → RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 각 모듈에 프레임 전달 → 결과 추론
        face_result = mp_face.process(frame_rgb)
        pose_result = mp_pose.process(frame_rgb)
        hands_result = mp_hands.process(frame_rgb)

        # 얼굴이 인식된 프레임 수 세기
        if face_result.multi_face_landmarks:
            face_detected += 1

        # 자세 또는 손동작이 감지된 프레임 수 세기
        if pose_result.pose_landmarks or hands_result.multi_hand_landmarks:
            gesture_detected += 1

        frame_count += 5 # 5프레임마다 체크(분석 속도 최적화를 위함) => 향후 pytorch기반 전환 고려

    cap.release()

    # 결과 통계 반환
    return {
        "total_frames": frame_count,
        "face_detected_frames": face_detected,
        "gesture_detected_frames": gesture_detected,
        "face_detection_ratio": round(face_detected / frame_count, 2) if frame_count else 0.0,
        "gesture_ratio": round(gesture_detected / frame_count, 2) if frame_count else 0.0,
    }


# ✅ 테스트용 실행 블록
if __name__ == "__main__":
    import sys
    import os

    # 기본 테스트 영상 경로
    default_video = "uploads/sample_video.mp4"

    # CLI 인자 or 기본 경로
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video

    # 파일 존재 여부 확인
    if not os.path.exists(video_path):
        print(f"영상 파일이 존재하지 않습니다: {video_path}")
    else:
        print(f"분석 중인 영상 파일: {video_path}\n")
        result = analyze_visual_features(video_path)

        # 결과 출력
        print("시각 표현 분석 결과:")
        for k, v in result.items():
            print(f"- {k}: {v}")
