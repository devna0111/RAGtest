from utils.video_processor import extract_audio, transcribe_audio, summarize_transcript
from utils.audio_analysis import analyze_audio_features
from utils.pose_analysis import analyze_visual_features
from utils.feedback_generator import generate_feedback

video_path = "uploads/your_video.mp4"
audio_path = extract_audio(video_path)

transcript = transcribe_audio(audio_path)
summary = summarize_transcript(transcript)
audio_stats = analyze_audio_features(audio_path)
visual_stats = analyze_visual_features(video_path)

feedback = generate_feedback(summary, audio_stats, visual_stats)
print(feedback)