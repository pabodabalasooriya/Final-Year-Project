import cv2
from model import load_model
import numpy as np 
#from scipy.misc import imresize
from skimage.transform import resize
from test import mean_squared_loss
from keras.models import load_model
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('modelpath',type=str)
args=parser.parse_args()
modelpath=args.modelpath

cap =cv2.VideoCapture(0)#, cv2.CAP_DSHOW
#rval=True
print('Loading model')
model=load_model(modelpath)
print('Model loaded')

threshold=0.1
while True:
	imagedump=[]
	for i in range(10):
		rval,frame=cap.read()
		frame=resize(frame,(227,227,3))
		#cv2.showVideo('frame', frame )
		

		#Convert the Image to Grayscale
		gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
		gray=(gray-gray.mean())/gray.std()
		gray=np.clip(gray,0,1)
		imagedump.append(gray)

	imagedump=np.array(imagedump)
	imagedump.resize(227,227,10)
	imagedump=np.expand_dims(imagedump,axis=0)
	imagedump=np.expand_dims(imagedump,axis=4)

	print('Processing data')
	output=model.predict(imagedump)
 
	loss=mean_squared_loss(imagedump,output)
 
	

	if loss>threshold:
		print('Anomalies Detected')

	#cv2.waitKey(0)

