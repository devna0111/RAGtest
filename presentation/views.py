from django.shortcuts import render

def presentation_analysis(request):
    return render(request, 'presentation/analysis.html')