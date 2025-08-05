from django.shortcuts import render

# Create your views here.
def search_page(request):
    return render(request, 'search/search.html')

def result(request) :
    if request.method == "POST" :
        return render(request, "search/search.html", context={"msg" : "check"})