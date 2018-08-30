# Furniture Classifier
Furniture Image Classifier using Keras (iMaterialist Challenge)

The following project was developed for the [iMaterialist Challenge (Furniture) at FGVC5](https://www.kaggle.com/c/imaterialist-challenge-furniture-2018), which consisted in developing a model to classify images into 128 different classes, the training dataset consisted of 194,828 images, the validation dataset consisted of 6,400 images and the testing dataset consisted of 12,800 images.

This project served me as a way to practice Deep Learning applied to image recognition, the model developed achieved a 32.86% error and scored 251th place out of 436 participants.

## About the model
I experimented using different model architectures and training strategies, due to time and hardware constraints the best performance/training_time results that I could obtain were using a Xception model with imagenet pre-trained weights, I retrained the last 80 layers of the model to get better results. 

The output layer has 128 units with a softmax activation function, categorical crossentropy was used as a loss function and I used a Nadam optimizer with a learning rate that was fine tuned during training (started at 0.001 and got smaller by a factor of 10 until reaching 0.0000001).

## Prequisites

The project uses Python 3, instructions to install it for every platform can be found at [python.org](https://www.python.org/).

This project requires [Tensorflow](https://www.tensorflow.org/). You can install with:
```
pip3 install --upgrade tensorflow
```
or
```
pip3 install --upgrade tensorflow-gpu
```
to install the GPU version.

The project also requires [Keras](https://keras.io/), install it with:
```
pip install keras
```

The project uses [Tqdm](https://github.com/tqdm/tqdm) to handle progress bars, install it with:
```
pip install tqdm
```

Te project uses [Matplotlib](https://matplotlib.org/) to plot the model's performance charts, to install use:
```
pip install matplotlib
```

The project uses [Numpy](http://www.numpy.org/), to install use:
```
pip install numpy
```

We use two libraries to manipulate images, [OpenCV](https://pypi.org/project/opencv-python/) and [imgaug](https://github.com/aleju/imgaug), these can be installed with:
```
pip install opencv-python
pip install imgaug
``` 

## Download images

The data folder contains a train and validation json, each json contains each image's id, url and label. The test json contains only the images' id and url.

The downloadImages.py script downloads the images in parallel (up to n threads) and stores them in the "img" folder, it receives three parameters: \[train/test/validation\]\(required\) \[startIndex\]\(optional\) \[jumps\]\(optional\), the first parameter determines the json that will be loaded and the output folder, the second parameter the index of the image to start downloading and the third parameter how many images to skip (this is because the dataset has a large size, so one may want to download only part of it).

Example of the downloadImages.py usage:
```
downloadImages.py train 1000 5
```
This command would start on image #1000 and skip 4 images each step \(1000, 1005, 1010, ..., etc.\).

## Create and train a model

The file neural.py can be used to train a model, it receives a model name as an input, if the model doesn't exist, it creates a model and trains it for 3 cycles of 5 epochs each, then saves it and outputs each cycle's performance plot in the plots folder. If the model already exists then it loads the model and perform 3 cycles of 5 epochs each, by doing this a model can be trained for any number of epochs.
Since there is a very large number of images, each cycle takes 2048 random images from the training set and trains the model with those images.

 ## Future work

* Set up ```install_requires``` to facilitate instalation of prequisites.
* Make the process of using different model architechtures easier, right now this is only available through editing the source code.

