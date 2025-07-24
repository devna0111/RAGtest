from flask import Flask, render_template, request, jsonify
from extract_utils import extract_text_and_chunks
import os

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/upload', methods=["GET",'POST'])
def upload():
    if request.method == "POST" :
        try:
            if 'file' not in request.files:
                return jsonify({'filename': None, 'chunks': [], 'error': "❌ 파일이 첨부되지 않았습니다."})
            file = request.files['file']
            if file.filename == '':
                return jsonify({'filename': None, 'chunks': [], 'error': "❌ 파일명이 비어 있습니다."})

            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # 👉 청크 추출
            chunks = extract_text_and_chunks(save_path)
            return jsonify({'filename': filename, 'chunks': chunks})
        except Exception as e:
            return jsonify({'filename': None, 'chunks': [], 'error': str(e)})
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
