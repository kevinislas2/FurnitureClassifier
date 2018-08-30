import sys
import numpy as np
import os
import random
import cv2 
from tqdm import tqdm
from model_utilities import loadOrCreateModel
filesPerClass = 10

def cropCenter(image):
	# print("IMage", image.shape)
	imgY, imgX, imgC = image.shape
	minShape = min(imgX, imgY)
	startX = imgX // 2 - minShape // 2
	startY = imgY // 2 - minShape // 2

	return image[startY:startY+minShape, startX:startX+minShape]


def createBatch(folder, imgClass, model):

	x = np.zeros((filesPerClass, 299, 299, 3))

	imgFolder = "{}/{}".format(folder, imgClass)
	dirF = os.listdir(imgFolder)
	# # paralellize this
	with tqdm(total=filesPerClass) as bar:
		for i in range(filesPerClass):
			imgName = random.choice(dirF)
			img = "{}/{}/{}".format(folder, imgClass, imgName)
			x[i] = cv2.resize(cropCenter(cv2.imread(img, cv2.IMREAD_UNCHANGED)), (299,299), interpolation  = cv2.INTER_AREA) / 255.
			bar.update(1)

	predY = model.predict(x)
	correct = 0
	for val in predY:
		if np.argmax(val)+1 == imgClass:
			correct+=1
	return correct
def main():
	if(len(sys.argv) != 2):
		print("argv[1] can't be None")
		exit()

	

	model = loadOrCreateModel(sys.argv[1])
	matrix = []
	for i in range(1, 129):
		matrix.append([i, createBatch("./images/validation", i, model)])

	matrix = sorted(matrix, key=lambda x: x[1])
	print(matrix)


if __name__ == '__main__':
	main()
