from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

def chat_page(request):
    return render(request, 'chat/chat.html')

@csrf_exempt
def file_upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if uploaded_file:
            # 파일 처리 로직 (나중에 구현)
            return JsonResponse({'status': 'success', 'filename': uploaded_file.name})
        return JsonResponse({'status': 'error', 'message': 'No file uploaded'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})