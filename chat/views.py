from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from chatbot_core import chatbot  # 추가
from vectordb_upload_search import data_to_vectorstore
import json
import os

def chat_page(request):
    return render(request, 'chat/chat.html')

@csrf_exempt
def file_upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if uploaded_file:
            # 임시 파일 저장
            file_path = f"temp/{uploaded_file.name}"
            os.makedirs('temp', exist_ok=True)
            with open(file_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            qdrant = data_to_vectorstore(file_path)
            # chatbot 함수 호출
            
            return JsonResponse({
                'status': 'success', 
                'filename': uploaded_file.name,
                'summary': "파일이 업로드 중입니다."  # 추가
            })
        else :
            file_path = "sample_inputs/sample.txt"
        return JsonResponse({'status': 'error', 'message': 'No file uploaded'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def chat_send(request):
    if request.method == 'POST':
        body = json.loads(request.body)
        message = body.get('message')
        # 여기에 AI 응답 생성 로직 추가
        answer = f"'{message}'에 대한 답변입니다!"
        return JsonResponse({'answer': answer})
    return JsonResponse({'error': 'Invalid method'}, status=405)