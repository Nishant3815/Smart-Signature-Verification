from django.http import JsonResponse#, HttpResponse, Http404
from django.shortcuts import render

from django.core.files import File
from sig_verify import accountSignatureExtraction

from sig_verify import signet
import cv2
from sig_verify import Database

import base64
from io import BytesIO
from PIL import Image
from scipy.misc import imread, imresize, imsave

from keras.models import model_from_json
import os


# Create your views here.
def main_page(request):
	return render(request, 'sig_verify/index1.html')
	
def data_return(request):
	if request.method == 'POST':
		accountNumber1 = request.POST.get('accountNumber')
		
		# Saving received image into Images as FormImage.png
		image1 = request.POST.get('image1')
		image1 = Image.open(BytesIO(base64.b64decode(image1[22:])))
		imsave('Images/FormImage.png', image1)
		#imsave('sig_verify/static/Images/FormImage.png', image1)
		
		# Sending FormImage.png to accountSIgnatureExtraction to get account number and sign image
		X = accountSignatureExtraction.acc_numo('Images/FormImage.png')
		#[full_name,acc_num,start_date,end_date,sign]
		signImage = X[4]
		
		print("Full_Name "+X[0])
		print("Account_Number "+str(X[1]))
		print("Start_Date  "+X[2])
		print("End_Date  "+X[3])
		
		# Database
		#Database.Add_Entry(12345678911, "sign.jpg")
		filename = Database.Extract(X[1])
		xx = filename[0]
		print (xx)
		# Saving signature image to images as SignImage.png
		imsave('Images/SignImage.png', signImage)
		
		# Sending SignImage to signet to get probability
		probability = signet.signet_classifier('Images/SignImage.png', 'Signatures/' + filename[0])
		print (probability)
		#x = "{% static '" + filename[0] + "' %}"
		# returning probability 
		data = {'account': str(X[1]), 'prob': str(probability), 'sig1':xx}
		print("save success!")
		#os.remove( r"C:\Users\HARSH\Desktop\FinalApp\mysite\Images\FormImage.png")
		return JsonResponse(data)
	else:
		data = {'is_taken': "NotAPostRequest" }
		return JsonResponse(data)
		

    
