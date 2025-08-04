from pptx import Presentation
import os
from langchain_ollama import ChatOllama
import pptx
from pptx.util import Pt, Inches
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
import re

def make_text_to_slide_text(whole_text:str) -> str :
    llm = ChatOllama(model="qwen2.5vl:7b")
    prompt = f"""
    당신은 발표안을 토대로 발표 슬라이드를 구성하는 전문가입니다.
    다음과 같은 발표안 [발표자료]를 토대로
    아래 양식 [참고자료]와 동일하게 ppt용 슬라이드 구성을 만들어서 return하세요.
    [발표자료]
    {whole_text}
    [참고자료]
    [슬라이드 1]
    제목: 사내 업무 자동화 소개
    핵심 포인트:
    - 업무 효율화 필요성
    - RAG 기반 자동화

    [슬라이드 2]
    제목: 주요 기능
    핵심 포인트:
    - 문서 요약 및 퀴즈 생성
    - 발표 자료 자동 작성
    - 벡터DB 기반 검색

    [슬라이드 3]
    제목: 기대 효과
    핵심 포인트:
    - 교육자료 제작 시간 단축
    - 구성원 이해도 향상
"""
    return llm.invoke([prompt]).content

from pptx import Presentation
from pptx.util import Pt, Inches
import re

def save_structured_text_to_pptx(whole_text,output_path: str = "output/presentation.pptx") -> str:
    structured_text = make_text_to_slide_text(whole_text)
    # 1. 프레젠테이션 객체 생성 + 16:9 비율 설정
    prs = Presentation()
    prs.slide_width = Inches(13.33)  # 16:9 비율
    prs.slide_height = Inches(7.5)

    slide_layout = prs.slide_layouts[1]  # 제목 + 내용

    # 2. 슬라이드 블록 파싱
    slide_blocks = re.split(r"### \[슬라이드 \d+\]", structured_text)
    slide_blocks = [b.strip() for b in slide_blocks if b.strip()]

    for block in slide_blocks:
        # 제목 추출
        title_match = re.search(r"#### 제목:\s*(.+)", block)
        title = title_match.group(1).strip() if title_match else "제목 없음"

        # 핵심 포인트 추출
        points = re.findall(r"-\s+(.+)", block)

        # 3. 슬라이드 생성
        slide = prs.slides.add_slide(slide_layout)

        # 제목 상자 커스터마이징
        title_shape = slide.shapes.title
        title_shape.text = title

        # 👉 슬라이드 전체 폭으로 제목 상자 확장
        title_shape.left = Inches(0.3)
        title_shape.top = Inches(0.3)
        title_shape.width = Inches(12.7)
        title_shape.height = Inches(1.5)

        # 👉 제목 폰트 크기 조정
        title_paragraph = title_shape.text_frame.paragraphs[0]
        title_paragraph.font.size = Pt(32)  # 필요 시 조정 가능

        # 본문 포인트 추가
        content_frame = slide.placeholders[1].text_frame
        content_frame.clear()

        for point in points:
            p = content_frame.add_paragraph()
            p.text = point
            p.level = 0
            p.font.size = Pt(18)

    prs.save(output_path)
    return output_path

if __name__ == "__main__":
    sample_text = """
안녕하세요, 여러분. 오늘은 서울 PM 수요예측 및 재배치 수요예측 모델 개발 프로젝트에 대해 발표드리려고 합니다. 이 프로젝트는 조명환, 정선우, 정종혁, 김도현 씨가 함께 수행한 빅데이터 활용 분석 모델 개발 프로젝트입니다.

먼저, 프로젝트의 개요를 살펴보겠습니다. 이 프로젝트는 개인형 이동장치(PM)의 수요 대비 공급 불균형 문제를 해결하기 위해 시작되었 습니다. 특히, 스마트 시티 트렌드와 서울 2024플랜에서 First/Last-Mile ↔ PM ↔ 대중교통 통합 네트워크가 요구되는 상황에서, 서울시  공유 이동수단의 배치 효율성을 높이기 위한 것입니다.

프로젝트의 목적은 서울시 공유 이동수단의 배치 효율성을 제고하는 것입니다. 이를 위해 실수요 기반 재배치 모델 및 정책 시뮬레이션  구축, 민간사업 진입 타당성 자료 확보 및 도시교통망의 효율화를 목표로 합니다.

프로젝트의 추진 전략 및 세부 목표는 다음과 같습니다. 데이터 기반 수요예측 모델 개발, 실시간 정보 활용, 정책 실험 기반 확장성 확 보 등이 포함됩니다.

프로젝트 추진 내용은 예측 모델 설계 및 구현으로 나뉩니다. 1단계 모델은 중장기 이용 패턴을 예측하기 위해 2022-2024년 따릉이 이용 이력을 대상으로 기온, 요일, 행정구역, 계절성 등 변수를 활용하여 LightGBM, LSTM, DNN 등의 알고리즘을 사용합니다. 2단계 모델은 초 단기 예측을 위해 실시간 유동인구, 기상, 시간대를 활용하여 자동 데이터 수집 코드를 구현합니다. 3단계 모델은 가상 PM 데이터를 결합하여 PM 수요 및 결입 횟수 예측을 수행하고, 이를 통해 정책 시뮬레이션을 수행할 수 있습니다.

프로젝트의 향후 일정은 다음과 같습니다. 7월 8일부터 10일까지는 데이터 수집 및 정제, 11일부터 12일까지는 탐색적 분석, 13일부터 15일까지는 M1 장기예측, 16일부터 18일까지는 M2 초단기예측, 19일부터 21일까지는 M3 통합 및 시나리오, 22일부터 23일까지는 웹 대시보 드 구축, 24일부터 25일까지는 문서 및 발표 준비가 예정되어 있습니다.

프로젝트의 리스크 및 대응 방안은 실증 데이터 부족, 법적 제약, 도로정보 불균형 등이 있습니다. 이에 대해 시뮬레이션 기반 검증 및  장기 데이터 확보, 제도화 시점 이전까지 가상 정책 실험 중심 운영, 향후 실측조사 병행 추진 계획 등을 통해 대응할 계획입니다.      

프로젝트의 기대 효과 및 활용 방안은 행정 효율 향상, 정책 자료화, 데이터 기반 행정 등이 있습니다. 이를 통해 수요 기반 공유 이동수단 운영으로 시민 불편 해소, 재배치 기준 정립 및 민간 사업 유치 근거 마련, 스마트 시티 구현을 위한 도시 데이터 기반 확립 등이 가 능할 것입니다.

마지막으로, 부록에 모델링 파이프라인과 시스템 아키텍처를 설명하겠습니다. 모델링 파이프라인은 데이터 인gest, 머신러닝 모델링, 재 배치 최적화 과정을 포함하고 있으며, 시스템 아키텍처는 데이터의 수집, 처리, 저장, 분석 등의 과정을 보여줍니다.

이러한 프로젝트를 통해 서울 PM 수요예측 및 재배치 수요예측 모델이 완성되면, 서울시 공유 이동수단의 효율적인 배치와 정책 시뮬레이션을 통해 도시 교통망의 효율화와 스마트 시티 구현에 기여할 수 있을 것입니다. 감사합니다.
"""
    # output_path = save_outline_to_pptx(sample_outline, "output/sample_presentation.pptx")
    # print(f"PPTX 저장 완료: {output_path}")
    test = make_text_to_slide_text(sample_text)
    print(test)
    save_structured_text_to_pptx(test)
