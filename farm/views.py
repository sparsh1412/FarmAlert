from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
	return render(request,'farm/home.html')

def about(request):
	return render(request,'farm/about.html')

def contact(request):
	return render(request,'farm/contact.html')



