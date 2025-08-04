# --------------------------
# 4. 간단한 보고서 작성
# --------------------------

def generate_report(chunks: List[Dict]) -> str:
    """추출된 청크 데이터를 기반으로 보고서 초안 작성"""
    report = "# DOCX 보고서 초안\n\n"
    report += "## 서론\n이 보고서는 DOCX 문서 내용을 기반으로 자동 생성되었습니다.\n\n"
    report += "## 본문\n"

    for c in chunks:
        if c["type"] == "text":
            report += f"- {c['content']}\n"
        elif c["type"] == "table":
            report += f"\n[표 데이터]: {c['content'][:200]}...\n"
        elif c["type"] == "image":
            report += f"\n[이미지 설명]: {c['content']}\n"

    report += "\n## 결론\n위 데이터를 기반으로 분석 및 요약을 진행할 수 있습니다.\n"
    return report