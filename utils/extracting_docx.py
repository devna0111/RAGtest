import os
from typing import Union
from docx import Document as DocumentLoader
from docx.document import Document
from docx.text.paragraph import Paragraph
from docx.table import _Cell, Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from multiprocessing import Process, Queue

from utils.image_utils import analyze_image_with_qwen  # 멀티모달 분석기

def iter_block_items(parent: Union[Document, _Cell]):
    """docx 문서 내 텍스트(paragraph)와 표(table)를 순서대로 순회"""
    parent_elm = parent._element.body if isinstance(parent, Document) else parent._element
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P): # paragraph
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl): # table
            yield Table(child, parent)

def analyze_worker(image_path, queue, mode : str = "simple"):
    """서브 프로세스에서 이미지 분석 실행"""
    print("scv 준비완료")
    try:
        result = analyze_image_with_qwen(image_path,mode=mode)
        queue.put((image_path, result.strip()))
    except Exception as e:
        queue.put((image_path, f"[에러] {e}"))

def parallel_image_analysis(image_paths: list) -> dict:
    """여러 이미지를 병렬로 분석"""
    queue = Queue()
    processes = []
    for path in image_paths:
        p = Process(target=analyze_worker, args=(path, queue))
        print("scv 업무 분장!")
        processes.append(p)
        p.start()

    results = {}
    for _ in processes:
        img_path, summary = queue.get()
        results[img_path] = summary

    for p in processes:
        p.join()

    return results

def extract_docx_content(docx_path: str, mode: str = "simple") -> str:
    """docx에서 텍스트, 표, 이미지(병렬 분석 포함)를 추출"""
    image_output_dir = "temp_imgs"
    os.makedirs(image_output_dir, exist_ok=True)

    doc = DocumentLoader(docx_path)
    content_list = []

    # 텍스트 및 표 추출
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                content_list.append(f"[텍스트]\n{text}")
                print("text 추출 중")
        elif isinstance(block, Table):
            rows = []
            for row in block.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                rows.append(row_text)
                print("표 추출 중")
            content_list.append(f"[표]\n" + "\n".join(rows))

    # 이미지 추출
    rels = doc.part._rels
    img_paths = []
    for idx, rel in enumerate(rels.values(), 1):
        if "image" in rel.reltype:
            img_data = rel.target_part.blob
            img_path = os.path.join(image_output_dir, f"image_{idx}.png")
            print("img 추출 중")
            with open(img_path, "wb") as f:
                f.write(img_data)
            img_paths.append(img_path)

    # 병렬로 이미지 분석
    img_summaries = parallel_image_analysis(img_paths)

    for idx, path in enumerate(img_paths, 1):
        print("qwen 작동")
        content_list.append(f"[이미지{idx}] 분석 결과:\n{img_summaries[path]}")

    return "\n\n".join(content_list)


# 테스트용
if __name__ == "__main__":
    docx_path = "sample_inputs/sample.docx"
    result = extract_docx_content(docx_path)
    print("전체 문서 추출 결과:\n", result)