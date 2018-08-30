import urllib.request
from tqdm import tqdm
import json
import sys, os, urllib3
from multiprocessing.pool import ThreadPool
import multiprocessing
from PIL import Image
from io import BytesIO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

data_path = "./data"
file = sys.argv[1]

test = False

if sys.argv[1] == "test":
	test = True
http = urllib3.PoolManager()

def downloadImage(img):
	if(test):
		img["label_id"] = "test"

	if not os.path.exists("./images/{}/{}".format(file, img["label_id"])):
		os.mkdir("./images/{}/{}".format(file, img["label_id"]))

	if not os.path.exists("./images/{}/{}/{}.jpg".format(file, img["label_id"], img["image_id"])):
		try:
			global http
			response = http.request("GET", img["url"][0])

			image_data = response.data
			pil_image = Image.open(BytesIO(image_data))
			pil_image_rgb = pil_image.convert('RGB')
			pil_image_rgb.save("./images/{}/{}/{}.jpg".format(file, img["label_id"], img["image_id"]), format='JPEG', quality=90)
		except:
			print("File {} failed".format(img["image_id"]))

def Run():
	
	files = ["train", "validation", "test"]
	jumps = 1
	assert file in files
	start = 0
	if(len(sys.argv) >= 3):
		start = int(sys.argv[2])

	if(len(sys.argv) == 4):
		jumps = int(sys.argv[3])
		
	imagesJson={}
	with open("{}/{}.json".format(data_path, file)) as json_data:
	    imagesJson= json.load(json_data)

	imagesJson["images"] = imagesJson["images"][start::jumps]
	if(not test):
		imagesJson["annotations"] = imagesJson["annotations"][start::jumps]
		for i in range(len(imagesJson["images"])):
			assert imagesJson["images"][i]["image_id"] == imagesJson["annotations"][i]["image_id"]
			imagesJson["images"][i]["label_id"] = imagesJson["annotations"][i]["label_id"]

	pool = multiprocessing.Pool(processes=8)
	with tqdm(total=len(imagesJson["images"])) as bar:
		for _ in pool.imap_unordered(downloadImage,  imagesJson["images"]):
			bar.update(1)

if __name__ == "__main__":
	files = []
	Run()