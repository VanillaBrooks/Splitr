import torch.utils.data
import pandas as pd
from skimage import io
import os
from build_training_data import OCR_DATA_PATH as training_data_path
from build_tensor import build_tensor_stack
import numpy as np
import torch

class OCR_dataset_loader(torch.utils.data.Dataset):
	def __init__(self, csv_file_path, path_to_data, encode=False, crop_dataset=False, transform=None):
		"""
		Arguments:
			csv_file_path (string): csv file with
			path_to_data (raw string): Directory holding all the data being used for training
			crop_dataset (int): crop the training data to a smaller size
			transform (callable, optional): Transformations to be applied to teh samples
		"""
		self.training_df = pd.read_csv(csv_file_path)
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
		if encode == 'onehot':
			image_text = encode_one_hot([image_text], len(self.unique_chars), self.unique_chars)
		elif encode = 'vector':
			image_text = encode_single_vector([image_text], len(self.unique_chars), self.unique_chars)

		return image , image_text

def find_unique_characters(list_to_parse):
	unique_chars = set()
	for word in list_to_parse:
		if isinstance(word,float):
			word = str(word)
		for letter in word:
			if letter not in unique_chars:
				unique_chars.add(letter)
	unique_chars = list(unique_chars)

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
	if not max_word_len:
		max_word_len = len(max(list_of_labels, key=len))
	# calculate the uniqe characters if not already done
	if not unique_chars:
		unique_chars = find_unique_characters(list_of_labels)

	tensors_to_stack = []

	for label in list_of_labels:
		if isinstance(label, float):
			# catch problems with NaN being in the dataset
			if str(label).lower() == 'nan':
				continue

		current_word_array = np.zeros((1,max_word_len), dtype=np.float32)
		array_placement_index = 0

		for letter in label:
			current_index = unique_chars.index(letter) # 0 will be blank
			current_word_array[0,array_placement_index] = current_index

			array_placement_index += 1

		# convert vector to tensor and save for later
		tensors_to_stack.append(torch.from_numpy(current_word_array).float())

	torch_stack = torch.stack(tensors_to_stack)
	return torch_stack

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
	x = OCR_dataset_loader(r'data\final.csv', r'C:\Users\Brooks\Desktop\OCR_data', None)
	print(x.unique_chars)
	a = encode_one_hot(x.training_df['labels'], x.max_str_len, x.unique_chars)
	print(a[1,:,:])
