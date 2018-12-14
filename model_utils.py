import torch.utils.data
import pandas as pd
from skimage import io
import os
from build_training_data import OCR_DATA_PATH as training_data_path
from build_tensor import build_tensor_stack
import numpy as np
import torch

class OCR_dataset_loader(torch.utils.data.Dataset):
	def __init__(self, csv_file_path, path_to_data, crop_dataset=False, transform=False):
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

		# if we are only working with a portion of the data loader
		if crop_dataset:
			self.training_df = self.training_df[:crop_dataset]

		# for encode decode
		self.char_count = 3
		self.characters = {'PAD':0}

		# get the maximum length of string in the data
		self.max_str_len = self.training_df.labels.map(lambda x: len(str(x))).max()
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

		full_img_path = os.path.join(self.root_dir,local_img_path)
		image = io.imread(full_img_path)	#numpy array of the image being fetched
		image = torch.from_numpy(image.astype(np.float32))

		# transform the image if needed
		if self.transform:
			image = self.transform(image)

		return image , image_text

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
	# print('the word vector is :')
	# print(word_vector.shape)
	# print(input_tensor.shape)
	vec = word_vector.squeeze()
	print (vec)

	print([unique_chars[int(i)] for i in vec])
	outstr = ''
	outstr = ''.join([unique_chars[int(i)] for i in vec if int(i) != 0])
	print(outstr)
	return outstr


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
