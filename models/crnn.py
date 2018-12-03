import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os
import time
import pandas as pd
from build_tensor import build_tensor_stack
from build_training_data import PATH as training_data_path

# https://arxiv.org/pdf/1306.2795v1.pdf
class model(torch.nn.Module):
	def __init__(self, channel_count=1):
		super(model, self).__init__()

		# add batch norm later
		# add dropout to cnn layers
		self.a = torch.nn.Conv2d(1,64,2)

		# CONVOLUTIONS
		self.cnn1 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=channel_count,out_channels=64,kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride=2)
		) # out: torch.Size([3, 64, 39, 249])

		self.cnn2 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((2,2), stride=2)
		) # out: torch.Size([3, 128, 18, 123])

		self.cnn3 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
		torch.nn.ReLU(),
		) # out: torch.Size([3, 256, 8, 60])

		self.cnn4 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
		torch.nn.ReLU(),
		torch.nn.MaxPool2d((1,2), stride = 2)
		) # out: torch.Size([3, 512, 3, 29])

		self.cnn5 = torch.nn.Sequential(
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2),
		torch.nn.ReLU(),
		torch.nn.BatchNorm2d(512),
		torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2)
		) # out:torch.Size([x, 512, 2, 28])

		# RECURRENT
		self.lstm1 = torch.nn.LSTM(input_size=224, hidden_size=112, num_layers=4, bidirectional=True)

		self.lstm2 = torch.nn.LSTM(input_size=224, hidden_size=112, num_layers=4, bidirectional=True)

		self.lin1 = torch.nn.Linear(in_features=512*224, out_features=16)

		self.r = torch.nn.ReLU()

		# self.lstm2 = torch.nn.Sequential(
		# torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2),
		# torch.nn.Linear(in_features=512*56, out_features=16)
		# )

	def forward(self, x):
		# q = lambda x: print(x.shape)
		t = self.cnn1(x)
		t = self.cnn2(t)
		t = self.cnn3(t)
		t = self.cnn4(t)
		t = self.cnn5(t)


		t = t.view(t.shape[0], -1, 224) # lstm input would be 512

		t , hidden = self.lstm1(t)
		t = self.r(t)
		# returns output, hidden
		# output is fed into the next network
		# t = t.view(t.shape[0], -1,256)


		t, hidden = self.lstm2(t)
		t = self.r(t)

		t = t.view(t.shape[0], -1)
		t = self.lin1(t)
		t = self.r(t)

		return t


def train(epochs=10000):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device being used: %s' % device)

	OCR = model().to(device)

	criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
	optimizer = torch.optim.SGD(OCR.parameters(), lr=1e-1)#.to(device)

	ct = 23
	debug = True

	data =pd.read_csv('final.csv')
	labels = data['labels'].tolist()
	filenames = data['names'].tolist()

	print('!!! initial length of labels and filenames: %s %s' % (len(labels), len(filenames)))
	# if we are debugging and ct is small we crop the data
	if debug and ct < len(labels):
		labels = labels[:ct]
		filenames = filenames[:ct]


	# make a vector representation of every word
	print('length of the data going into one hot %s' % len(labels))
	training_data_y = one_hot_lables(labels).to(device)

	training_images = []
	s = time.time()
	i = 1

	# add all the files to a list
	for path in filenames:
		im = cv.imread(os.path.join(training_data_path, path))
		im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
		training_images.append(im)

		if i % 10 ==0:
			print('importing images: %s' % (i*100 / ct))
		i += 1

	print('datda going into build tensor stack: %s ' % len(training_images))
	training_data_x = build_tensor_stack(training_images).to(device)


	for i in range(epochs):
		optimizer.zero_grad()
		# print(':::::::::::::training datax %s training data y %s' % (len(training_data_x), len(training_data_y)))
		predicted_vals = OCR.forward(training_data_x)

		loss = criterion(predicted_vals.squeeze(), training_data_y.squeeze())
		loss.backward()

		if i % 10 == 0:
			print('epoch:%s loss %s'% (i, loss.item()))


def one_hot_lables(list_of_labels):
	with open('unique_characters.txt', 'r') as f:
		chars = f.read()

	np_arrays_to_stack = []

	for label in list_of_labels:
		current_word_array = np.zeros((1,16), dtype=np.float32)
		array_placement_index = 0

		label = str(label)

		for letter in label:
			current_index = chars.index(letter) +1 # let zero be blank
			current_word_array[0,array_placement_index] = current_index

			array_placement_index += 1

		# convert vector to tensor and save for later
		np_arrays_to_stack.append(torch.from_numpy(current_word_array).float())

	torch_stack = torch.stack(np_arrays_to_stack)
	return torch_stack


if __name__ == '__main__':
	train()

	# df = pd.read_csv('final.csv')
	# x = df['labels'].tolist()
	# print(x)
