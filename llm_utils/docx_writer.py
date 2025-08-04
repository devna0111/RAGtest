from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn
import re

def markdown_to_styled_docx(markdown_text: str, output_path: str = "output/project.docx") -> str:
    doc = Document()

    for line in markdown_text.splitlines():
        line = line.strip()
        if not line:
            doc.add_paragraph()
            continue

        # 제목 스타일
        if line.startswith("### "):  # H3
            clean_line = line.replace("### ", "").strip()
            para = doc.add_paragraph()
            run = para.add_run(clean_line)
            run.bold = True
            run.font.size = Pt(13)

        elif line.startswith("#### "):  # H4
            clean_line = line.replace("#### ", "").strip()
            para = doc.add_paragraph()
            run = para.add_run(clean_line)
            run.bold = True
            run.font.size = Pt(11)

        else:
            # 일반 문장 처리 + 굵은 텍스트 감지 (예: **굵게**)
            para = doc.add_paragraph()
            segments = re.split(r"(\*\*[^\*]+\*\*)", line)

            for segment in segments:
                run = para.add_run()
                run.font.size = Pt(10)
                run.font.name = '맑은 고딕'
                run._element.rPr.rFonts.set(qn('w:eastAsia'), '맑은 고딕')

                if segment.startswith("**") and segment.endswith("**"):
                    run.bold = True
                    run.text = segment[2:-2]
                else:
                    run.bold = False
                    run.text = segment

    doc.save(output_path)
    return output_path


if __name__ == "__main__":
    from pathlib import Path

    markdown_text = """### 서울 PM 수요예측 & 재배치 수요예측 모델 기획서

#### 1. 개요
서울 PM 수요예측 & 재배치 수요예측 모델 개발 프로젝트는 서울시 공유 이동수단의 배치 효율성을 높이고, 실시간 정보 기반 재배치 모델 및 정책 시뮬레이션을 구축 하여 도시교통망의 효율화를 목표로 진행된다.

#### 2. 주요 내용
- **목적**: 서울시 공유 이동수단의 배치 효율성 제고
- **구성요소**: 실시간 정보 기반 재배치 모델 및 정책 시뮬레이션 구축, 민간사업 진입 타당성 자료 확보 및 도시교통망의 효율화
- **추진 전략**: 데이터 기반 수요예측, 실시간 정보 활용, 정책 실험 기반 확장성 확보

#### 3. 세부 목표
- **데이터 기반 수요예측**: 장/단기 이용패턴 예측 모델 개발
- **실시간 정보 활용**: 유동인구 및 기상 등 실시간 정보 기반 초단기 예측
- **정책 실험 기반 확장성 확보**: 가상 PM 데이터 반영 및 정책 실험 가능 모델 설계

#### 4. 프로젝트 추진 내용
- **예측 모델 설계 및 구현**
  - **1단계 모델 (중장기)**: 2022-2024년 따릉이 이용 이력, 기온, 요일, 행정구역, 계절성 등, LightGBM, LSTM, DNN 등 알고리즘
  - **2단계 모델 (초단기)**: 실시간 유동인구, 기상, 시간대, 자동 데이터 수집 코드 구현 (API 활용)
  - **3단계 모델**: 가상 PM 데이터를 결합해 PM 수요 및 결입 횟수 예측, 정책 시뮬레이션 수행 가능

#### 5. 데이터 구축 및 수집
- **서울시 열린데이터 기반 따릉이 대여 이력**
- **기상청 날씨 정보 및 자전거 도로 인프라**
- **실시간 유동인구 API 연동**
- **가상 PM 운영정보(시나리오 기반 설정)**

#### 6. 향후 일정 (안)
- **7/08 – 07/10**: 데이터 수집 & 정제 (CSV 적재, 결측치·이상치 처리)
- **7/11 – 07/12**: 탐색적 분석 (EDA) (인사이트 그래프, 피처 리스트)
- **7/13 – 07/15**: M1 장기예측 (LightGBM/LSTM 결과 + 리포트)
- **7/16 – 07/18**: M2 초단기예측 (TFT/XGBoost 결과 + 리포트)
- **7/19 – 07/21**: M3 통합·시나리오 (재배치 알고리즘 + 테스트 케이스)
- **7/22 – 07/23**: 웹 대시보드 (PoC) (Streamlit 배포 URL 생성 및 테스트)
- **7/24 – 07/25**: 문서 & 발표준비 (기획서 PDF + PPT 완성)

#### 7. 리스크 및 대응 방안
- **실증 데이터 부족**: 시뮬레이션 기반 검증 및 장기 데이터 확보 추진
- **법적 제약**: 제도화 시점(2025년) 이전까지 가상 정책 실험 중심 운영
- **도로정보 불균형**: 향후 실측조사 병행 추진 계획 수립

#### 8. 기대 효과 및 활용 방안
- **행정효율 향상**: 수요 기반 공유 이동수단 운영으로 시민 불편 해소
- **정책자료화**: 재배치 기준 정립 및 민간 사업 유치 근거 마련
- **데이터 기반 행정**: 스마트 시티 구현을 위한 도시데이터 기반 확립

#### 9. 부록
- **모델링 파이프 라인**
- **시스템 아키텍처**

---

이 기획서는 서울 PM 수요예측 & 재배치 수요예측 모델 개발 프로젝트의 전반적인 개요와 구체적인 추진 계획을 담고 있으며, 이를 통해 서울시의 도시교통망 효율화와 스마트 시티 구현을 위한 데이터 기반 행정을 목표로 진행될 것이다.
"""

    Path("output").mkdir(exist_ok=True)
    docx_path = markdown_to_styled_docx(markdown_text)
    print(f"DOCX 생성 완료: {docx_path}")
