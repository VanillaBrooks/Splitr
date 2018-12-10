import torch.utils.data
import pandas as pd
from skimage import io
import os
from build_training_data import OCR_DATA_PATH as training_data_path
from build_tensor import build_tensor_stack
import numpy as np
import torch

class OCR_dataset_loader(torch.utils.data.Dataset):
	def __init__(self, csv_file_path, path_to_data, crop_dataset=False, transform=None):
		"""
		Arguments:
			csv_file_path (string): csv file with
			path_to_data (raw string): Directory holding all the data being used for training
			crop_dataset (int): crop the training data to a smaller size
			transform (callable, optional): Transformations to be applied to teh samples
		"""
		self.training_df = pd.read_csv(csv_file_path)
		if crop_dataset:
			self.training_df = self.training_df[:crop_dataset]
		self.root_dir = path_to_data
		self.transform = transform
		self.char_count = 3
		self.characters = {'PAD':0}

	def __len__(self):
		return len(self.training_df)

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

	def text2tensor(self):
		pass



if __name__ == '__main__':
	x = OCR_dataset_loader(r'data\final.csv', r'C:\Users\Brooks\Desktop\OCR_data', None)
	# print(x.__len__())
	print(x.__getitem__(0))
