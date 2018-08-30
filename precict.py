from model_utilities import saveModel, loadOrCreateModel, saveHistory
import numpy as np
from os.path import isfile
from random import randrange
import sys
from tqdm import tqdm
import tensorflow as tf
import os
import cv2
def cropCenter(image):
	# print("IMage", image.shape)
	imgY, imgX, imgC = image.shape
	minShape = min(imgX, imgY)
	startX = imgX // 2 - minShape // 2
	startY = imgY // 2 - minShape // 2

	return image[startY:startY+minShape, startX:startX+minShape]

def main():
	if(len(sys.argv) != 2):
		print("argv[1] can't be None")
		exit()

	print("START")

	with tf.Session() as sess:
		model = loadOrCreateModel(sys.argv[1])

		
		with open("pred.csv", "w") as file:

			file.write("id,predicted\n")
			for i in range(1, 12801):
			# for i in range(89809, 90488):
				imageName = "images/test/test/{}.jpg".format(i)
				# imageName = "images/train/1/{}.jpg".format(i)
				if(isfile(imageName)):
					image = cv2.resize(cropCenter(cv2.imread(imageName, cv2.IMREAD_UNCHANGED)), (299,299), interpolation  = cv2.INTER_AREA) / 255.
					image = np.expand_dims(image, axis=0)

					y = np.argmax(model.predict(image))+1
					# print(image[0], y )
					file.write("{},{}\n".format(i, y))
				else:
					print("ERR")
					file.write("{},{}\n".format(i, randrange(0,127)))
			

if __name__ == '__main__':
	main()