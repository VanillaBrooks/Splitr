import torch.utils.data
import pandas as pd
from skimage import io, transform
import os
from build_training_data import OCR_DATA_PATH as training_data_path
import numpy as np
import torch
import torchvision
import random
import cv2 as cv
import math


class OCR_dataset_loader(torch.utils.data.Dataset):
	def __init__(self, csv_file_path, path_to_data, transform):
		"""
		Arguments:
			csv_file_path (string): csv file with
			path_to_data (raw string): Directory holding all the data being used for training
			encode (onehot / vector): Decides what type of data to return from __getitem__
				if left empty a tuple of strings will be returned
			crop_dataset (int): crop the training data to a smaller size
			transform (callable, optional): Transformations to be applied to teh samples
		"""
		self.training_df = pd.read_csv(csv_file_path).dropna()
		self.root_dir = path_to_data
		self.transform = transform

		# get the maximum length of string in the data
		self.max_str_len = self.training_df.labels.map(lambda x: len(str(x))).max()
		# print('the max str len is %s'% self.max_str_len)
		self.training_df_len = len(self.training_df)

		# generate a list of unique characters of this dataset
		self.unique_chars = find_unique_characters(self.training_df['labels'])


	def save_unique_chars(self, path=r'data\unique_chars.txt'):
		with open(path, 'w') as f:
			f.write(''.join(self.unique_chars))

	def __len__(self):
		return self.training_df_len

	def __getitem__(self, idx):
		row_of_csv = self.training_df.iloc[idx, :]

		# organize the data
		image_text= row_of_csv[0]
		local_img_path = row_of_csv[1]
		local_img_path = local_img_path[:-6] + ' ' + local_img_path[-6:]	# take care of the space that was generated in the actual files' names REMOVE LATER

		full_img_path = os.path.join(self.root_dir,local_img_path)
		image = io.imread(full_img_path).astype(np.float32)

		# transform the image if needed
		if self.transform:
			image = self.transform(image)

		# image = np.asarray(image)
		# cv.imwrite(os.path.join(r'C:\Users\Brooks\Desktop\test', local_img_path), image)

		if isinstance(image, np.ndarray):
			image= torch.from_numpy(image)
		image = image[None, : , : ]

		return image, image_text

class Rotate():
	def __init__ (self, max_angle):
		self.angle = max_angle

	def __call__(self, image):
		angle = random.uniform(-1*self.angle, self.angle)
		image = transform.rotate(image, angle, resize=True, clip=False, preserve_range=True, cval=255)

		return image

class Pad():
	def __init__(self):
		self.DESIRED_WIDTH = 250
		self.DESIRED_HEIGHT = 40

	def __call__(self, image):
		calculate_padding = lambda desired, actual: desired-actual

		dims = image.shape
		height, width = dims[0], dims[1]

		h_ratio,w_ratio,ratio = 0, 0 ,0

		debug = 'initial height was {}, initial width was {} \n'.format(height,width)


		if height > self.DESIRED_HEIGHT:
			h_ratio = height / self.DESIRED_HEIGHT
		if width > self.DESIRED_WIDTH:
			w_ratio = width / self.DESIRED_WIDTH


		if h_ratio > w_ratio:
			ratio = h_ratio
		elif w_ratio >= h_ratio:
			ratio = w_ratio

		debug += 'h_ratio is {} w_ratio is {} \n result ratio is {}\n'.format(h_ratio, w_ratio, ratio)

		if ratio:
			height= math.floor(height / ratio)
			width = math.floor(width / ratio)

			debug = 'resized height was {}, resized width was {} \n'.format(height,width)

			image = cv.resize(image, dsize=(width, height), interpolation=cv.INTER_CUBIC)


		padding_height = calculate_padding(self.DESIRED_HEIGHT, height)
		padding_width = calculate_padding(self.DESIRED_WIDTH, width)

		pad_top, pad_bot, pad_left, pad_right = 0 ,0, 0, 0

		if padding_height > 0:
			pad_top = random.randint(0, padding_height)
			pad_bot = padding_height - pad_top

		if padding_width > 0:
			pad_left = random.randint(0, padding_width)
			pad_right = padding_width - pad_left

		debug += 'padding height is {} padtop: {} pad bot {} \npadding width is {} pad_left {} pad right {}'.format(padding_height, pad_top, pad_bot, padding_width, pad_left,pad_right)


		pad_function = torch.nn.ConstantPad2d((pad_left, pad_right, pad_top,pad_bot),255)
		result = pad_function(torch.from_numpy(image).float())

		height,width = result.shape
		debug += 'final height {} final width {}'.format(height, width)

		if height != self.DESIRED_HEIGHT or width != self.DESIRED_WIDTH:
			print(debug)
			raise ValueError('dims not correctly done in Pad')

		return result


def find_unique_characters(list_to_parse):
	unique_chars = list()
	for word in list_to_parse:
		if isinstance(word,float):
			word = str(word)
		for letter in word:
			if letter not in unique_chars:
				unique_chars.append(letter)

	# make sure that space is at the start of the character sequence
	if ' ' not in unique_chars:
		unique_chars = [' '] + unique_chars
	elif ' ' in unique_chars:
		if unique_chars[0] != ' ':
			unique_chars.pop(unique_chars.index(' '))
			unique_chars = [' '] + unique_chars

	return unique_chars

def encode_single_vector(list_of_labels, max_word_len=False, unique_chars=False):
	# calculate maxlen if not already done
	# print('max word len: ', max_word_len)
	if not max_word_len:
		max_word_len = len(max(list_of_labels, key=len))
	# calculate the uniqe characters if not already done
	if not unique_chars:
		unique_chars = find_unique_characters(list_of_labels)

	tensors_to_stack = []

	for label in list_of_labels:
		if isinstance(label, float):
			# catch problems with NaN being in the dataset
			# if str(label).lower() == 'nan':
				# continue
			print('problem label: ', label)
			label = str(label)

		current_word_array = np.zeros((1,max_word_len), dtype=np.float32)
		# print('current word array:')
		# print(current_word_array.shape)
		array_placement_index = 0

		for letter in label:
			current_index = unique_chars.index(letter) # 0 will be blank
			current_word_array[0,array_placement_index] = current_index

			array_placement_index += 1

		# convert vector to tensor and save for later
		tensors_to_stack.append(torch.from_numpy(current_word_array).float())

	torch_stack = torch.stack(tensors_to_stack)
	return torch_stack

def decode_single_vector(input_tensor, unique_chars, argmax_dim=1):
	word_vector = torch.argmax(input_tensor, argmax_dim)
	print(unique_chars)
	vec = word_vector.squeeze()
	print (vec)

	characters  = [unique_chars[int(i)] for i in vec]
	print(len(characters), characters)
	raw_outstr = ''
	format_outstr = ''

	previous_character = ' '
	for index in vec:
		character = unique_chars[int(index)]

		if character == ' ':
			raw_outstr += character
			if previous_character == ' ':
				pass
			else:
				# raw_outstr += character
				format_outstr += character
		elif character != previous_character:
			raw_outstr += character
			format_outstr += character
		else:
			raw_outstr += character
		previous_character = character
	return (raw_outstr, format_outstr)

def encode_one_hot(list_of_labels, max_word_len=False, unique_chars=False):
	# calculate max_word_len if not already done
	if not max_word_len:
		max_word_len = len(max(list_of_labels, key=len))
	# calculate the uniqe characters if not already done
	if not unique_chars:
		unique_chars = find_unique_characters(list_of_labels)

	unique_chars_len= len(unique_chars)
	tensors_to_stack = []

	for label in list_of_labels:
		if isinstance(label, float):
			# catch problems with NaN being in the dataset
			if str(label).lower() == 'nan':
				continue
			label = str(label)
		current_word_array = np.zeros((max_word_len,unique_chars_len), dtype=np.float32)

		array_placement_index = 0
		for letter in label:
			letter_index = unique_chars.index(letter)
			current_word_array[array_placement_index, letter_index] = 1
			array_placement_index+=1

		tensors_to_stack.append(torch.from_numpy(current_word_array).float())
	torch_stack = torch.stack(tensors_to_stack)
	return torch_stack

if __name__ == '__main__':
	transforms = torchvision.transforms.Compose([Rotate(5), Pad()])

	training_set = OCR_dataset_loader(
		csv_file_path = r'C:\Users\Brooks\github\Splitr\data\training_data.csv',
		path_to_data  = r'C:\Users\Brooks\Desktop\OCR_data',
		transform = transforms)

	training_data = torch.utils.data.DataLoader(
		training_set,
		batch_size=1,
		num_workers=0,
		shuffle=0)
