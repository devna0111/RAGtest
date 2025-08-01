import os
import threading
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

import document_processor_test
from rag_pipeline_test import RAGPipeline

# --- Flask 및 시스템 설정 ---
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DB_PATH = 'chroma_db_multimodal'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 사용할 모델 이름 설정
TEXT_MODEL = "gemma3:1b"
EMBED_MODEL = "bge-m3:567m" # 임베딩도 동일 모델 사용

# RAG 파이프라인 인스턴스 생성
rag_system = RAGPipeline(DB_PATH, TEXT_MODEL, EMBED_MODEL)

# --- 백그라운드 처리 함수 ---
def process_and_add_to_db(filepath):
    """백그라운드에서 문서 처리 및 DB 추가를 수행합니다."""
    try:
        # 1. 파일에서 텍스트/이미지 설명 추출
        texts = document_processor_test.process_document(filepath)
        
        # 2. 추출된 텍스트를 벡터 DB에 추가
        if texts:
            rag_system.add_texts_to_db(texts)
        else:
            print(f"'{filepath}'에서 처리할 텍스트를 찾지 못했습니다.")
            
    except Exception as e:
        print(f"백그라운드 처리 중 오류 발생: {e}")

# --- Flask 라우트 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '파일이 선택되지 않았습니다.'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 백그라운드 스레드에서 문서 처리 시작
        thread = threading.Thread(target=process_and_add_to_db, args=(filepath,))
        thread.start()

        return jsonify({
            'status': 'success', 
            'message': f"'{filename}' 파일 업로드 성공! 문서 처리가 백그라운드에서 시작되었습니다."
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'파일 업로드 중 오류 발생: {e}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('question')

    if not query:
        return jsonify({'answer': '질문을 입력해주세요.'})

    try:
        result = rag_system.query(query)
        
        # 디버깅을 위해 소스 문서 출력
        print("\n--- 참고한 소스 문서 ---")
        for doc in result.get('source_documents', []):
            print(f"- 내용: {doc.page_content[:150]}...")
        print("---------------------\n")
        
        return jsonify({'answer': result.get('result', '답변을 생성하지 못했습니다.')})

    except Exception as e:
        print(f"채팅 처리 중 오류 발생: {e}")
        return jsonify({'answer': f'답변 생성 중 오류가 발생했습니다: {e}'}), 500

# --- 앱 실행 ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

