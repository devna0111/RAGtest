from utils.image_utils import analyze_image_with_qwen

if __name__ == "__main__":
    import time
    path = "temp_imgs/image_1.png"
    t1 = time.time()
    result = analyze_image_with_qwen(path)
    t2 = time.time()

    print("\n[단일 분석 테스트] 소요 시간:", round(t2 - t1, 2), "초")
    print(result[:500])
