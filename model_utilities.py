from keras.models import model_from_json
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, SeparableConv2D, MaxPooling2D, Flatten
from keras.applications.resnet50 import ResNet50
from os.path import isfile
import os
from keras.optimizers import RMSprop, Adam, Adagrad, Adamax, Nadam
from matplotlib import pyplot as plt
from time import time
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
freeze_layers = 80
lr = 0.0000001

def saveModel(model, modelName):
	modelJson = "{}.json".format(modelName)
	modelH5 = "{}.h5".format(modelName)
	print("saveModel")
	model_json = model.to_json()
	with open(modelJson, "w") as json_file:
		json_file.write(model_json)
	#seralize weights to HDF5
	model.save_weights(modelH5)
	print("Saved model to disk")

def loadModel(modelName):
	print("loadModel")
	modelJson = "{}.json".format(modelName)
	modelH5 = "{}.h5".format(modelName)
	json_file = open(modelJson, "r")
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	#load weights
	model.load_weights(modelH5)

	# for layer in model.layers:
	# 	layer.trainable = True

	for layer in model.layers[:-freeze_layers]:
		layer.trainable = False
	for layer in model.layers[-freeze_layers:]:
		layer.trainable = True

	model.compile(loss="categorical_crossentropy", optimizer=Nadam(lr=lr), metrics=["categorical_accuracy"])
	# model.summary()
	return model

def saveHistory(f1, val_f1, modelName, epoch, jumps):
	plt.plot(f1)
	plt.plot(val_f1)
	plt.title("Model accuracy: Epochs {}-{}".format(epoch,epoch+jumps))
	plt.ylabel("acc")
	plt.xlabel("epoch")
	plt.legend(["train", "test"], loc="upper left")
	if(not os.path.exists("./plots/{}".format(modelName))):
		os.mkdir("./plots/{}".format(modelName))
	plt.savefig("./plots/{}/{}.png".format(modelName, epoch))
	plt.clf()
def createModel():

	# return MyModel()
	input_shape = (299, 299, 3)

	tf_input = Input(shape=input_shape)
	# base_model = ResNet50(include_top=False, weights="imagenet", input_tensor=tf_input, pooling="max")
	base_model = Xception(include_top=False, weights="imagenet", input_tensor=tf_input, pooling="max")
	# base_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=tf_input, pooling="max")
	# base_model = InceptionV3(include_top=False, weights="imagenet", input_tensor=tf_input, pooling="max")
	# for layer in base_model.layers:
	# 	layer.trainable = False
	top = base_model.output
	# top = Dense(2048, activation="relu")(top)
	top = Dense(128, activation="softmax")(top)
	model = Model(inputs=base_model.input, outputs=top)
	
	for layer in model.layers[:-freeze_layers]:
		layer.trainable = False
	for layer in model.layers[-freeze_layers:]:
		layer.trainable = True

	model.compile(loss="categorical_crossentropy", optimizer=Nadam(lr=lr), metrics=["categorical_accuracy"])
	model.summary()
	return model

def loadOrCreateModel(modelName):
	if(isfile("{}.json".format(modelName)) and isfile("{}.h5".format(modelName))):
		print("Loading model")
		return loadModel(modelName)
	else:
		print("Creating model")
		return createModel()