from keras.preprocessing.image import ImageDataGenerator
from model_utilities import saveModel, loadOrCreateModel, saveHistory
import numpy as np
import sys
import os, random
import cv2 
from tqdm import tqdm
from keras.utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from os.path import isfile
import pickle
from imgaug import augmenters as iaa

batchSize = 1024*2

modelParametersFile = r"./{}.pickle".format(sys.argv[1])
sometimes = lambda aug: iaa.Sometimes(0.3, aug)
seq = iaa.Sequential([
	# iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
	iaa.Fliplr(0.5), # horizontally flip 50% of the images
	# sometimes(iaa.Add((-.1, .1), per_channel=0.5)),
	# iaa.GaussianBlur(sigma=(0, 3.0)), # blur images with a sigma of 0 to 3.0
	# iaa.SomeOf((0, 5),
	# 	[
	# 		iaa.OneOf([
	# 				iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
	# 				iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
	# 				# iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
	# 		]),
	# 		iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
	# 		# iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
	# 		iaa.SimplexNoiseAlpha(iaa.OneOf([
	# 			iaa.EdgeDetect(alpha=(0.5, 1.0)),
	# 			iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
	# 		])),
	# 		# iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
	# 		# iaa.OneOf([
	# 			# iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
	# 			# iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
	# 		# ]),
	# 		# iaa.Add((-10, 10), per_channel=0.5),
	# 		# iaa.OneOf([
	# 		# 	iaa.Multiply((0.5, 1.5), per_channel=0.5),
	# 		# 	iaa.FrequencyNoiseAlpha(
	# 		# 		exponent=(-4, 0),
	# 		# 		first=iaa.Multiply((0.5, 1.5), per_channel=True),
	# 		# 		second=iaa.ContrastNormalization((0.5, 2.0))
	# 		# 	)
	# 		# ]),
		# ], random_order=True)
])

def load():
	modelP = None
	if(isfile(modelParametersFile)):
		print("Load P")
		with open(modelParametersFile, "rb") as input_pickle:
			modelP = pickle.load(input_pickle)
			modelP["jumps"] = 5
	else:
		print("Creating P")
		modelP = {"epochs": 0, "jumps": 5}
	return modelP

def save(p):
	with open(modelParametersFile, "wb") as output_pickle:
		pickle.dump(p, output_pickle)

def cropCenter(image):
	# print("IMage", image.shape)
	imgY, imgX, imgC = image.shape
	minShape = min(imgX, imgY)
	startX = imgX // 2 - minShape // 2
	startY = imgY // 2 - minShape // 2

	return image[startY:startY+minShape, startX:startX+minShape]

def getRandomClass(dirF):
	# To handle unbalance in predictions
	# if(random.random() <=.10):
	# 	return 63
	# elif(random.random() <=.10):
	# 	return 70
	# elif(random.random() <=.10):
	# 	return 8
	# else:
	return int(random.choice(dirF))
		# return 63
		# return 87
		# return 15
		# return 4
		# return 66
		# return 70
		# return 8

def createBatch(batchSize, folder, aug):
	x = np.zeros((batchSize, 299, 299, 3))
	y = np.zeros((batchSize, 128))
	dirF = os.listdir(folder)

	# paralellize this
	with tqdm(total=batchSize) as bar:
		for i in range(batchSize):
			imgLabel = getRandomClass(dirF)
			y[i] =  to_categorical(imgLabel - 1, 128)
			f = "{}/{}".format(folder,imgLabel)
			img = "{}/{}".format(f, random.choice(os.listdir(f)))
			x[i] = cv2.resize(cropCenter(cv2.imread(img, cv2.IMREAD_UNCHANGED)), (299,299), interpolation  = cv2.INTER_AREA)
			if(np.random.uniform()>=0.5):
				x[i] = cv2.flip(x[i], 0)
			x[i] = x[i] / 255.
			bar.update(1)
	if(aug):
		x = seq.augment_images(x)
	return x, y
	
def main():
	if(len(sys.argv) != 2):
		print("argv[1] can't be None")
		exit()

	trainX, trainY = createBatch(batchSize, "./images/train", False)
	valX, valY = createBatch(512, "./images/validation", False)

	model = loadOrCreateModel(sys.argv[1])

	modelP = load()
	e = modelP["epochs"] + modelP["jumps"]
	history = model.fit(trainX, trainY, batch_size=2, epochs=e, validation_data=(valX, valY), initial_epoch=modelP["epochs"])

	saveModel(model, sys.argv[1])
	saveHistory(history.history["categorical_accuracy"], history.history["val_categorical_accuracy"], sys.argv[1], modelP["epochs"], modelP["jumps"])
	modelP["epochs"] += modelP["jumps"]
	save(modelP)

	# return history.history["categorical_accuracy"], history.history["val_categorical_accuracy"]

def Run():
	f1Arr = []
	valF1Arr = []
	for i in range(3):
		main()
		
if __name__ == '__main__':
	Run()
