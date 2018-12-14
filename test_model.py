from models import crnn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv
import torch
import model_utils
import skimage
import sys
import time

def make_image_array(input_word,save=False):
	# construct tensor holding the optical characters
	img = Image.new('L', (500,80), color=255)
	imdraw= ImageDraw.Draw(img)
	font = ImageFont.truetype(FONT_PATH,15)
	imdraw.text((250, 40), TEST_WORD, fill=0, font=font)
	np_image = np.asarray(img)
	input_tensor = torch.from_numpy(np_image[None, None,:,:]).float()

	if save:
		cv.imwrite('%s.png' % input_word, np_image)

	return input_tensor

def load_training_images(path):
	image = skimage.io.imread(path)[None,None,:,:]
	input_tensor = torch.from_numpy(image).float()

	return input_tensor

FONT_PATH = r'C:\Users\Brooks\github\Splitr\fonts\OpenSans-Bold.ttf'# the path to the font that is being used
MODEL_LOAD_PATH = r'C:\Users\Brooks\github\Splitr\models\1544758832_%s_%s.model' % (1,11600)# path to the model being loaded and tested
LOAD_MODEL = True
TEST_WORD = 'example text' # phrase being drawn in picture

img = make_image_array(input_word=TEST_WORD, save=True)

# initialize model and load weights
model = crnn.model(channel_count=1,num_hidden= 256, unique_char_count=57,rnn_layer_stack=2)
if LOAD_MODEL:
	model.load_state_dict(torch.load(MODEL_LOAD_PATH))
	t = time.time()
with torch.no_grad():
	word_result = model.forward(img)
print('model eval runtime: %s' % (time.time()- t))

# fetch the characterset being used in the model
dataloader = model_utils.OCR_dataset_loader(
		csv_file_path = r'C:\Users\Brooks\github\Splitr\data\training_data.csv',
		path_to_data =r'C:\Users\Brooks\Desktop\OCR_data',
		crop_dataset=False,
		transform = False)

char_set = dataloader.unique_chars
# word_len = dataloader.max_str_len

# find the resulting tensor and word 2
result_word = model_utils.decode_single_vector(word_result,char_set, 2)

print('input word was: %s' % TEST_WORD)
print('output word from model:', result_word)
