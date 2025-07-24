from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64
from extract_img import extract_image_chunks
# Ollama 인스턴스
llm = ChatOllama(model="gemma3:4b")  # 모델명에 :4b, :12b 등 태그 주의 granite3.2-vision:2b
# granite, gemma3:12b, gemma3:4b, LLaVA 중 gemma3가 그 중 가장 적당한 결과 반환

# 이미지 base64 인코딩
with open("sample5.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")
text = extract_image_chunks("sample5.png")
# 프롬프트와 이미지를 HumanMessage에 담기
msg = HumanMessage(
    content=[
        {"type": "text", "text": "이미지 속 텍스트 또는 표형태의 json형태로 반환해주세요. 없는 내용은 추가하지 않으며 인식한 내용만 반영합니다."},
        # {"type": "text", "text": f"다음은 OCR을 통해 추출한 내용을 정리한 것입니다. 참고자료로 활용해주세요.{text}"},
        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
    ]
)

# 결과 받기
response = llm.invoke([msg])
print(response.content)
print(text)
print('gemma 후처리결과')
print(llm.invoke(f"""다음은 OCR을 통해 확인한 text입니다.
                이를 좀 더 자연스럽게 보완해주세요.
                [OCR로 확인한 text]{text}
                추가적인 정보는 요하지 않으며 json형식으로 반환합니다."""))
