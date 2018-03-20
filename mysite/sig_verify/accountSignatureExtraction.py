import os
import cv2
#import matplotlib.pyplot as plt
from keras.models import model_from_json
import numpy as np
#from pytesseract import image_to_string
from scipy import ndimage
import sklearn
#from skimage.transform import resize
from sklearn.externals import joblib
import keras.backend.tensorflow_backend

import tensorflow as tf

def acc_numo(st):

	if keras.backend.tensorflow_backend._SESSION:
		tf.reset_default_graph()
		keras.backend.tensorflow_backend._SESSION = None

	form=cv2.imread(st,0)
	retval, thresh_gray = cv2.threshold(form, thresh=220, maxval=255, type=cv2.THRESH_BINARY)
	points = np.argwhere(thresh_gray==0) # find where the black pixels are
	points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
	x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
	x, y, w, h = x, y, w, h # make the box a little bigger
	form = form[y:y+h, x:x+w]
	crop=cv2.resize(form, (1992,1000),interpolation=cv2.INTER_CUBIC)
	#acc
	[x, y, w, h] = [470, 220, 580, 70]
	acc = crop[y:y+h,x:x+w]
	#sign
	[x, y, w, h] = [620, 680, 300, 150]
	sign = crop[y:y+h,x:x+w]
	def sort_contours(cnts):
		boundingBoxes = [cv2.boundingRect(c) for c in cnts]
		(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][0]))
		 # return the list of sorted contours and bounding boxes
		return (cnts, boundingBoxes)
	json_file = open('MNIST_Weights/model_MNIST.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("MNIST_Weights/model_MNIST.h5")
	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	#account number
	acc_num=0
	ret, mask = cv2.threshold(acc, 220, 255, cv2.THRESH_BINARY)
	mask=cv2.erode(mask,(3,3),iterations=2)
	_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	cnts,bx=sort_contours(contours)
	for contour in cnts:
		[x, y, w, h] = cv2.boundingRect(contour)
		if w>40 and h>50:
			if w<80:
				new_img=mask[y+1:y+h-1,x+1:x+w-1]
				new_img=cv2.resize(new_img, (28,28),interpolation=cv2.INTER_CUBIC)
#             new_img=cv2.erode(new_img, (3, 3),iterations=1)
				new_img=cv2.dilate(new_img,(3,3),iterations=1)
				new_img=cv2.bitwise_not(new_img)
				ans=loaded_model.predict(new_img.reshape(1,28,28,1)).tolist()[0]
				acc_num=acc_num*10+(ans.index(max(ans)))
#             plt.imshow(new_img,cmap='gray')
#             plt.show()


	#start_date
	[x, y, w, h] = [120, 720, 430, 60]
	date_s= crop[y:y+h,x:x+w]
	start_date=""
	ret, mask = cv2.threshold(date_s, 200, 255, cv2.THRESH_BINARY)
	mask=cv2.erode(mask,(3,3),iterations=1)
	_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts,bx=sort_contours(contours)
	for contour in cnts:
		[x, y, w, h] = cv2.boundingRect(contour)
		if w>40 and h>50:
			if w<80:
				new_img=mask[y+1:y+h-1,x+1:x+w-1]
				new_img=cv2.resize(new_img, (28,28),interpolation=cv2.INTER_CUBIC)
				new_img=cv2.dilate(new_img,(3,3),iterations=1)
				new_img=cv2.bitwise_not(new_img)
				ans=loaded_model.predict(new_img.reshape(1,28,28,1)).tolist()[0]
				start_date+=str(ans.index(max(ans)))
	start_date=start_date[:2]+"-"+start_date[2:4]+"-"+start_date[4:]


	#end_date
	[x, y, w, h] = [120, 790, 430, 60]
	date_e= crop[y:y+h,x:x+w]
	end_date=""
	ret, mask = cv2.threshold(date_e, 200, 255, cv2.THRESH_BINARY)
	mask=cv2.erode(mask,(3,3),iterations=1)
	_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts,bx=sort_contours(contours)
	for contour in cnts:
		[x, y, w, h] = cv2.boundingRect(contour)
		if w>40 and h>50:
			if w<80:
				new_img=mask[y+1:y+h-1,x+1:x+w-1]
				new_img=cv2.resize(new_img, (28,28),interpolation=cv2.INTER_CUBIC)
				new_img=cv2.dilate(new_img,(3,3),iterations=1)
				new_img=cv2.bitwise_not(new_img)
				ans=loaded_model.predict(new_img.reshape(1,28,28,1)).tolist()[0]
				end_date+=str(ans.index(max(ans)))
	end_date=end_date[:2]+"-"+end_date[2:4]+"-"+end_date[4:]

	#amount
	[x, y, w, h] = [1655, 370, 350, 50]
	amt= crop[y:y+h,x:x+w]
	# amount=image_to_string(amt,config='outputbase digits')

	#name
	clf = joblib.load("MNIST_Weights/MLP.pkl")
	[x, y, w, h] = [470, 150, 800, 60]
	name= crop[y:y+h,x:x+w]

	def get_labels(crop):
		img = crop.copy() # gray-scale image
		threshold = 0.8
		labeled, nr_objects = ndimage.label(img<threshold)
		#print("Number of objects is " +str(nr_objects))
		return labeled, nr_objects
	def get_bboxes(labeled, nr_objects):
		bboxes = np.zeros((nr_objects, 2, 2), dtype='int')
		x1, y1, x2, y2 = 0, labeled.shape[0], 0, 0
		coord = 0
		cont = 0
		ytop, ybot = 0, 0
		nzero, firstb = False, False
		for x in range(0, labeled.shape[1]):
			nzero, firstb = False, False
			ytop, ybot = 0, 0
			for y in range(0, labeled.shape[0]):
				if (labeled[y][x] > 0):
					nzero = True
					if (not firstb):
						ytop = y
						firstb = True
					ybot = y

			if (nzero):
				if (ytop < y1):
					y1 = ytop
				if (ybot > y2):
					y2 = ybot
				if (coord == 0):
					x1 = x
					coord = 1
				elif (coord == 1):
					x2 = x
			elif ((not nzero) and (coord == 1)):
				bboxes[cont][0] = [x1, y1]
				bboxes[cont][1] = [x2, y2]
				cont += 1
				coord = 0
				x1, y1, x2, y2 = 0, labeled.shape[0], 0, 0

		bboxes = bboxes[0:cont]
		return bboxes, cont
	def crop_characters(img, bboxes, n):
		characters = []
		for i in range(0, n):
			c = img.copy()[bboxes[i][0][1]:bboxes[i][1][1], bboxes[i][0][0]:bboxes[i][1][0]]
			if (c.shape[0] != 0 and c.shape[1] != 0):
				c = resize(c, (28, 28), mode='constant', cval=1.0, clip=True)
				characters.append((c<0.80).reshape(784))
		return characters, len(characters)
	def get_characters_img_only(image):
		labeled, nr_objects = get_labels(image)
		bboxes, n = get_bboxes(labeled, nr_objects)
		characters, n_chars = crop_characters(image, bboxes, n)
		return characters

	name=cv2.fastNlMeansDenoising(name,10,7,21)
	ret, mask = cv2.threshold(name, 180, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	mask=cv2.fastNlMeansDenoising(mask,10,7,21)

	t=get_characters_img_only(mask)
	full_name=""
	for i in range(len(t)):
		n=t[i].reshape(28,28)
		if (cv2.countNonZero(np.array(n,dtype=np.uint8))!=784):
			full_name+=clf.predict(n.reshape(1,-1))[0]

	print(str(acc_num))
	return [full_name,acc_num,start_date,end_date,sign]
