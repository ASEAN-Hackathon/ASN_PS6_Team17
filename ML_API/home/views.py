from django.shortcuts import render
from .models import *
from PIL import Image
from .forms import *
from .function import *
from PIL import Image
# Create your views here.
import requests
from io import BytesIO

def home(request):
    return render(request,'home.html')

def register(request):
    if request.method == 'POST':
        url = request.POST['fname']
        response = requests.get(str(url),stream = True)
        with open('media/images/out.jpg', 'wb') as f:
	        f.write(response.content)
       #img = Image.open(requests.get(url, stream=True).raw)
        #img.save('media/results/1.jpg')
        dict = final()
        return render(request,'result.html',{'dict':dict})
            
    else:  
        return render(request,'register.html')

