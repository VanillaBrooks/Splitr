from models import crnn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv
import torch
import model_utils


# the path to the font that is being used
FONT_PATH = r'C:\Users\Brooks\github\Splitr\futurefonts\OpenSans-Italic.ttf'
# path to the model being loaded and tested
MODEL_LOAD_PATH = r'C:\Users\Brooks\github\Splitr\models\1544605757_90.model'
# phrase being drawn in picture
TEST_WORD = 'this is a sample'

# construct tensor holding the optical characters
img = Image.new('L', (500,80), color=255)
imdraw= ImageDraw.Draw(img)
font = ImageFont.truetype(FONT_PATH,35)
imdraw.text((10,-10), TEST_WORD, fill=0, font=font)
np_image = np.array(img)[None, None,:,:]
input_tensor = torch.from_numpy(np_image).float()

# initialize model and load weights
model = crnn.model()
model.load_state_dict(torch.load(MODEL_LOAD_PATH))
with torch.no_grad():
	word_result = model.forward(input_tensor)


# fetch the characterset being used in the model
dataloader = model_utils.OCR_dataset_loader(
		csv_file_path = r'C:\Users\Brooks\github\Splitr\data\final.csv',
		path_to_data =r'C:\Users\Brooks\Desktop\OCR_data',
		crop_dataset=False,
		transform = False)
char_set = dataloader.unique_chars
word_len = dataloader.max_str_len
# print(char_set)

# find the resulting tensor and word
result_word = model_utils.decode_single_vector(word_result,char_set, 2)
