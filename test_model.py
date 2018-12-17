from models import crnn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv
import torch
import model_utils
import skimage
import sys
import time

def make_image_array(input_word,FONT_SIZE=15, save=False):
	# construct tensor holding the optical characters
	img = Image.new('L', (500,80), color=255)
	imdraw= ImageDraw.Draw(img)
	font = ImageFont.truetype(FONT_PATH,FONT_SIZE)
	imdraw.text((250, 30), TEST_WORD, fill=0, font=font)
	np_image = np.asarray(img)
	input_tensor = torch.from_numpy(np_image[None, None,:,:]).float()

	if save:
		cv.imwrite('%s.png' % input_word, np_image)

	return input_tensor


def load_training_images(path):
	image = skimage.io.imread(path)[None,None,:,:]
	input_tensor = torch.from_numpy(image).float()

	return input_tensor

def configure_device(device_mode):
	if device_mode != "gpu" and device_mode != 'cpu':
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	elif device_mode == 'gpu':
		device =torch.device('cuda:0')
	elif device_mode == 'cpu':
		if torch.cuda.is_available():
			print('warning: cpu was selected but CUDA is available')
		device = torch.device(device_mode)
	return device

def main(FONT_PATH, MODEL_PATH,LOAD_MODEL,TEST_WORD, csv_file_path, path_to_data, FONT_SIZE, device):
	a1 = time.time()
	img = make_image_array(input_word=TEST_WORD, FONT_SIZE=FONT_SIZE, save=True).to(device)

	a2 = time.time()
	# initialize model and load weights
	model = crnn.model(channel_count=1,num_hidden= 256, unique_char_count=57,rnn_layer_stack=2).to(device)
	if LOAD_MODEL:
		model.load_state_dict(torch.load(MODEL_LOAD_PATH))

	a3 = time.time()

	with torch.no_grad():
		word_result = model.forward(img)

	a4 = time.time()

	# fetch the characterset being used in the model
	dataloader = model_utils.OCR_dataset_loader(
			csv_file_path = csv_file_path,
			path_to_data =path_to_data,
			crop_dataset=False,
			transform = False)
	char_set = dataloader.unique_chars

	a5 = time.time()

	# find the resulting tensor and word 2
	result_word = model_utils.decode_single_vector(word_result,char_set,2)

	a6 = time.time()
	print('image input word: {} \nraw output from model: {} \nParsed output from model: {}\n'.format(TEST_WORD, *result_word))
	print('image array {} \nLoad Model {} \nEval Model {} \nDataloader {} \ndecode {} \nTotal: {}'.format(a2-a1,a3-a2,a4-a3,a5-a4,a6-a5, a6-a1))


if __name__ == '__main__':
	FONT_PATH = r'fonts\OpenSans-Bold.ttf'# the path to the font that is being used
	MODEL_LOAD_PATH = r'models/model.model'# path to the model being loaded and tested
	LOAD_MODEL = True
	TEST_WORD = 'sampletext' # phrase being drawn in picture

	csv_file_path = r'data\training_data.csv'
	path_to_data =r'C:\Users\Brooks\Desktop\OCR_data'

	FONT_SIZE = 15

	MODE = 'cpu'

	main(FONT_PATH,MODEL_LOAD_PATH,LOAD_MODEL,TEST_WORD,csv_file_path, path_to_data,FONT_SIZE, configure_device(MODE))
