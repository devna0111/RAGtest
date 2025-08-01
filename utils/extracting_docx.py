import os
from typing import Union
from docx import Document as DocumentLoader  # 파일 로드용
from docx.document import Document           # 타입 체크용
from docx.text.paragraph import Paragraph
from docx.table import _Cell, Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
# 이미지 분석 유틸 호출
from utils.image_utils import analyze_image_with_qwen

def iter_block_items(parent: Union[Document, _Cell]):
    """
    docx 문서 내 텍스트(paragraph)와 표(table)를 순서대로 순회
    """
    parent_elm = parent._element.body if isinstance(parent, Document) else parent._element
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def extract_docx_content(docx_path: str) -> str:
    """
    docx 파일에서 텍스트, 표, 이미지(그래프 포함)를 순서대로 하나의 문자열로 추출
    추출 결과는 향후 청크 분리/요약에 활용될 수 있도록 가공됨
    """
    image_output_dir="temp_imgs"
    os.makedirs(image_output_dir, exist_ok=True)
    doc = DocumentLoader(docx_path)
    content_list = []  # 최종 문자열 리스트

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                content_list.append(f"[텍스트]\n{text}")
        elif isinstance(block, Table):
            table_text = []
            for row in block.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                table_text.append(row_text)
            table_block = "\n".join(table_text)
            content_list.append(f"[표]\n{table_block}")

    # 이미지 처리
    rels = doc.part._rels
    img_idx = 1
    for rel in rels.values():
        if "image" in rel.reltype:
            img_data = rel.target_part.blob
            img_path = os.path.join(image_output_dir, f"image_{img_idx}.png")
            with open(img_path, "wb") as f:
                f.write(img_data)

            # 이미지 요약 수행
            summary = analyze_image_with_qwen(img_path)
            content_list.append(f"[이미지{img_idx}] Qwen 분석 결과:\n{summary.strip()}")
            img_idx += 1

    return "\n\n".join(content_list).strip()


if __name__ == "__main__":
    docx_path = "sample_inputs/sample.docx"
    result = extract_docx_content(docx_path)
    print("전체 문서 추출 결과:\n", result)
